using Microsoft.AspNetCore.Mvc;
using System.Net.Http.Headers;
using System.Text.Json;

namespace ChatBot.Server.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class ChatController : ControllerBase
    {
        private readonly HttpClient _httpClient;
        private readonly string _openAiApiKey;
        private readonly string _azureSearchApiKey;
        private readonly string _azureSearchService;
        private readonly string _indexName;

        public ChatController(IHttpClientFactory httpClientFactory, IConfiguration configuration)
        {
            _httpClient = httpClientFactory.CreateClient();
            _openAiApiKey = configuration["OpenAI:ApiKey"] ?? throw new ArgumentNullException("OpenAI API key missing");
            _azureSearchApiKey = configuration["AzureSearch:ApiKey"] ?? throw new ArgumentNullException("Azure Search API key missing");
            _azureSearchService = configuration["AzureSearch:ServiceUri"] ?? throw new ArgumentNullException("Azure Search URI missing");
            _indexName = configuration["AzureSearch:IndexName"] ?? throw new ArgumentNullException("Azure Search Index name missing");
        }

        [HttpPost("with-eshop-assistant")]
        public async Task<IActionResult> VectorSearch([FromBody] QuestionRequest request)
        {
            try
            {
                // 1️⃣ Get embedding from OpenAI
                var embeddingVector = await GetEmbeddingAsync(request.Question);

                // 2️⃣ Search Azure AI Search with embedding
                var topChunks = await SearchAzureAsync(embeddingVector);

                if (!topChunks.Any())
                    return NotFound("No relevant content found to answer the question.");

                // 3️⃣ Generate final answer using GPT with context
                var answer = await GenerateAnswerFromContextAsync(request.Question, topChunks);

                return Ok(new { Answer = answer });
            }
            catch (Exception ex)
            {
                return StatusCode(500, new { error = ex.Message });
            }
        }

        private async Task<List<float>> GetEmbeddingAsync(string question)
        {
            var embeddingRequest = new
            {
                input = question,
                model = "text-embedding-ada-002"
            };

            var openAiRequest = new HttpRequestMessage(HttpMethod.Post, "https://api.openai.com/v1/embeddings")
            {
                Content = JsonContent.Create(embeddingRequest)
            };
            openAiRequest.Headers.Authorization = new AuthenticationHeaderValue("Bearer", _openAiApiKey);

            var embeddingResponse = await _httpClient.SendAsync(openAiRequest);
            embeddingResponse.EnsureSuccessStatusCode();

            var embeddingData = await embeddingResponse.Content.ReadFromJsonAsync<EmbeddingResponse>();
            return embeddingData!.data[0].embedding;
        }

        private async Task<List<string>> SearchAzureAsync(List<float> embeddingVector)
        {
            var searchRequestBody = new
            {
                vectorQueries = new[]
                {
            new
            {
                kind = "vector",
                vector = embeddingVector,
                k = 5,
                fields = "embedding_vector_1"
            }
        },
                select = "*",
                top = 5
            };

            var searchUrl = $"{_azureSearchService}/indexes/{_indexName}/docs/search?api-version=2024-07-01";
            var searchRequest = new HttpRequestMessage(HttpMethod.Post, searchUrl)
            {
                Content = JsonContent.Create(searchRequestBody)
            };
            searchRequest.Headers.Add("api-key", _azureSearchApiKey);

            var searchResponse = await _httpClient.SendAsync(searchRequest);
            var searchJson = await searchResponse.Content.ReadAsStringAsync();

            if (!searchResponse.IsSuccessStatusCode)
                return new List<string>();

            var searchResults = JsonDocument.Parse(searchJson);
            return searchResults.RootElement
                .GetProperty("value")
                .EnumerateArray()
                .Select(doc => doc.GetProperty("chunk_text").GetString())
                .Where(text => !string.IsNullOrEmpty(text))
                .ToList();
        }

        private async Task<string> GenerateAnswerFromContextAsync(string question, List<string> contextChunks)
        {
            var context = string.Join("\n\n", contextChunks);

            var gptRequest = new
            {
                model = "gpt-3.5-turbo",
                messages = new[]
                {
            new { role = "system", content = "You are a helpful assistant. Use the provided context to answer the user's question." },
            new { role = "user", content = $"Context:\n{context}\n\nQuestion:\n{question}" }
        },
                temperature = 0.7
            };

            var gptApiRequest = new HttpRequestMessage(HttpMethod.Post, "https://api.openai.com/v1/chat/completions")
            {
                Content = JsonContent.Create(gptRequest)
            };
            gptApiRequest.Headers.Authorization = new AuthenticationHeaderValue("Bearer", _openAiApiKey);

            var gptResponse = await _httpClient.SendAsync(gptApiRequest);
            var gptResponseJson = await gptResponse.Content.ReadAsStringAsync();

            gptResponse.EnsureSuccessStatusCode();

            var gptResult = JsonDocument.Parse(gptResponseJson);
            return gptResult.RootElement
                .GetProperty("choices")[0]
                .GetProperty("message")
                .GetProperty("content")
                .GetString();
        }

    }



    public class QuestionRequest
    {
        public string Question { get; set; } = string.Empty;
    }

    public class EmbeddingResponse
    {
        public List<EmbeddingData> data { get; set; } = new();
    }

    public class EmbeddingData
    {
        public List<float> embedding { get; set; } = new();
    }
}
