using Microsoft.AspNetCore.Mvc;
using Azure.Storage.Blobs;
using Azure.Search.Documents;
using Azure.Search.Documents.Models;
using OpenAI;
using UglyToad.PdfPig;
using System.Text;
using Azure;
using System.Reflection;

[ApiController]
[Route("api/[controller]")]
public class DataPreparationController : ControllerBase
{
    private readonly string _blobConnectionString;
    private readonly string _containerName;
    private readonly string _searchServiceUri;
    private readonly string _indexName;
    private readonly string _searchApiKey;
    private readonly string _openAiApiKey;
    private readonly BlobContainerClient _blobContainerClient;
    private readonly SearchClient _searchClient;
    private readonly OpenAIClient _openAiApi;

    public DataPreparationController(IConfiguration configuration)
    {
        //Ideally use Key vault for storing these secrets instead of settings.json
        _blobConnectionString = configuration["AzureBlobStorage:ConnectionString"] ?? throw new ArgumentNullException("AzureBlobStorage:ConnectionString", "Blob storage connection string is missing.");
        _containerName = configuration["AzureBlobStorage:ContainerName"] ?? throw new ArgumentNullException("AzureBlobStorage:ContainerName", "Blob storage container name is missing.");
        _searchServiceUri = configuration["AzureSearch:ServiceUri"] ?? throw new ArgumentNullException("AzureSearch:ServiceUri", "Azure Search service URI is missing.");
        _indexName = configuration["AzureSearch:IndexName"] ?? throw new ArgumentNullException("AzureSearch:IndexName", "Azure Search index name is missing.");
        _searchApiKey = configuration["AzureSearch:ApiKey"] ?? throw new ArgumentNullException("AzureSearch:ApiKey", "Azure Search API key is missing.");
        _openAiApiKey = configuration["OpenAI:ApiKey"] ?? throw new ArgumentNullException("OpenAI:ApiKey", "OpenAI API key is missing.");

        _blobContainerClient = new BlobContainerClient(_blobConnectionString, _containerName);
        _searchClient = new SearchClient(new Uri(_searchServiceUri), _indexName, new AzureKeyCredential(_searchApiKey));
        _openAiApi = new OpenAIClient(_openAiApiKey);
    }

    [HttpPost("process-pdfs")]
    public async Task<IActionResult> ProcessPdfs()
    {
        var documents = new List<object>();

        await foreach (var blobItem in _blobContainerClient.GetBlobsAsync())
        {
            var blobClient = _blobContainerClient.GetBlobClient(blobItem.Name);

            using var stream = await blobClient.OpenReadAsync();
            using var pdf = PdfDocument.Open(stream);
            var totalPages = pdf.NumberOfPages;

            for (int pageNumber = 1; pageNumber <= totalPages; pageNumber++)
            {
                try
                {
                    var page = pdf.GetPage(pageNumber);
                    string pageText = page.Text;

                    var chunks = ChunkText(pageText);

                    foreach (var chunk in chunks)
                    {
                        var embedding = await GetEmbeddingAsync(chunk);

                        var document = new
                        {
                            id = Guid.NewGuid().ToString(),
                            chunk_text = chunk,
                            embedding_vector = embedding,
                            source_file = blobItem.Name,
                            page_number = pageNumber
                        };

                        documents.Add(document);
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error processing page {pageNumber} in {blobItem.Name}: {ex.Message}");
                    continue;
                }
            }

        }

        // Bulk upload to Azure AI Search
        await _searchClient.UploadDocumentsAsync(documents);

        return Ok(new { message = "Data preparation complete", totalDocuments = documents.Count });
    }

    private List<string> ChunkText(string text, int maxChunkSize = 1000)
    {
        var words = text.Split(' ');
        var chunks = new List<string>();
        var currentChunk = new StringBuilder();

        foreach (var word in words)
        {
            if (currentChunk.Length + word.Length + 1 > maxChunkSize)
            {
                chunks.Add(currentChunk.ToString());
                currentChunk.Clear();
            }
            currentChunk.Append(word + " ");
        }

        if (currentChunk.Length > 0)
            chunks.Add(currentChunk.ToString());

        return chunks;
    }

    private async Task<List<float>> GetEmbeddingAsync(string text)
    {
        var embeddingClient = _openAiApi.GetEmbeddingClient("text-embedding-ada-002");
        var result = await embeddingClient.GenerateEmbeddingAsync(text);
        if (result.Value == null)
        {
            throw new Exception("Failed to generate embedding.");
        }

        return result.Value.ToFloats().ToArray().Select(v => (float)v).ToList();
    }
}
