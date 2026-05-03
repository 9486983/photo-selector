using PhotoSelector.Application.Interfaces;
using PhotoSelector.Domain.Models;
using System.Net.Http.Json;
using System.Text.Json;

namespace PhotoSelector.Infrastructure.Services;

public sealed class HttpAiServiceClient(HttpClient httpClient) : IAiServiceClient
{
    private const int MaxRetries = 3;
    private const int RetryBaseDelayMs = 1000;

    public async Task<AnalyzeResponse> AnalyzeAsync(string imagePath, CancellationToken cancellationToken = default)
    {
        var payload = new { image_path = imagePath };

        for (var attempt = 0; attempt <= MaxRetries; attempt++)
        {
            try
            {
                using var response = await httpClient.PostAsJsonAsync("/analyze", payload, cancellationToken);
                response.EnsureSuccessStatusCode();

                var json = await response.Content.ReadAsStringAsync(cancellationToken);
                var root = JsonSerializer.Deserialize<JsonElement>(json);
                return ParseAnalyzeResponse(root, json);
            }
            catch (Exception ex) when (attempt < MaxRetries && IsRetryable(ex))
            {
                var delay = (int)(RetryBaseDelayMs * Math.Pow(2, attempt)) + Random.Shared.Next(0, 200);
                System.Diagnostics.Debug.WriteLine(
                    $"[HttpAiServiceClient] Analyze attempt {attempt + 1}/{MaxRetries} failed for {imagePath}: {ex.Message}. Retrying in {delay}ms");
                await Task.Delay(delay, cancellationToken);
            }
            catch (HttpRequestException ex) when (attempt < MaxRetries)
            {
                var delay = (int)(RetryBaseDelayMs * Math.Pow(2, attempt)) + Random.Shared.Next(0, 200);
                System.Diagnostics.Debug.WriteLine(
                    $"[HttpAiServiceClient] HTTP error attempt {attempt + 1}/{MaxRetries} for {imagePath}: {ex.Message}. Retrying in {delay}ms");
                await Task.Delay(delay, cancellationToken);
            }
        }

        // Final attempt - let exceptions propagate
        using var finalResponse = await httpClient.PostAsJsonAsync("/analyze", payload, cancellationToken);
        finalResponse.EnsureSuccessStatusCode();
        var finalJson = await finalResponse.Content.ReadAsStringAsync(cancellationToken);
        var finalRoot = JsonSerializer.Deserialize<JsonElement>(finalJson);
        return ParseAnalyzeResponse(finalRoot, finalJson);
    }

    public async Task<IReadOnlyCollection<(string ImagePath, AnalyzeResponse Response)>> AnalyzeBatchAsync(
        IReadOnlyCollection<string> imagePaths,
        CancellationToken cancellationToken = default)
    {
        var payload = new { image_paths = imagePaths.ToArray() };

        for (var attempt = 0; attempt <= MaxRetries; attempt++)
        {
            try
            {
                using var response = await httpClient.PostAsJsonAsync("/analyze/batch", payload, cancellationToken);
                if (!response.IsSuccessStatusCode)
                {
                    return await FallbackToIndividualAsync(imagePaths, cancellationToken);
                }

                var json = await response.Content.ReadAsStringAsync(cancellationToken);
                var root = JsonSerializer.Deserialize<JsonElement>(json);
                var result = new List<(string, AnalyzeResponse)>();

                if (root.TryGetProperty("items", out var items) && items.ValueKind == JsonValueKind.Array)
                {
                    foreach (var item in items.EnumerateArray())
                    {
                        if (!item.TryGetProperty("image_path", out var pathNode))
                        {
                            continue;
                        }

                        var path = pathNode.GetString() ?? string.Empty;
                        result.Add((path, ParseAnalyzeResponse(item, item.GetRawText())));
                    }
                }

                return result;
            }
            catch (Exception ex) when (attempt < MaxRetries && IsRetryable(ex))
            {
                var delay = (int)(RetryBaseDelayMs * Math.Pow(2, attempt)) + Random.Shared.Next(0, 200);
                System.Diagnostics.Debug.WriteLine(
                    $"[HttpAiServiceClient] Batch attempt {attempt + 1}/{MaxRetries} failed: {ex.Message}. Retrying in {delay}ms");
                await Task.Delay(delay, cancellationToken);
            }
        }

        return await FallbackToIndividualAsync(imagePaths, cancellationToken);
    }

    private async Task<IReadOnlyCollection<(string, AnalyzeResponse)>> FallbackToIndividualAsync(
        IReadOnlyCollection<string> imagePaths,
        CancellationToken cancellationToken)
    {
        var result = new List<(string, AnalyzeResponse)>();
        foreach (var imagePath in imagePaths)
        {
            result.Add((imagePath, await AnalyzeAsync(imagePath, cancellationToken)));
        }

        return result;
    }

    private static bool IsRetryable(Exception ex) => ex switch
    {
        HttpRequestException => true,
        TaskCanceledException => true,
        _ => false,
    };

    private static AnalyzeResponse ParseAnalyzeResponse(JsonElement root, string rawJson)
    {
        var result = new AnalyzeResponse
        {
            OverallScore = root.TryGetProperty("overall_score", out var overall) ? overall.GetSingle() : 0,
            SharpnessScore = root.TryGetProperty("sharpness_score", out var sharpness) ? sharpness.GetSingle() : 0,
            ExposureScore = root.TryGetProperty("exposure_score", out var exposure) ? exposure.GetSingle() : 0,
            EyesClosed = root.TryGetProperty("eyes_closed", out var eyesClosed) && eyesClosed.GetBoolean(),
            IsDuplicate = root.TryGetProperty("is_duplicate", out var duplicate) && duplicate.GetBoolean(),
            FaceCount = root.TryGetProperty("face_count", out var faceCount) ? faceCount.GetInt32() : 0,
            PersonLabel = root.TryGetProperty("person_label", out var personLabel) ? personLabel.GetString() ?? "none" : "none",
            StyleLabel = root.TryGetProperty("style_label", out var styleLabel) ? styleLabel.GetString() ?? "unknown" : "unknown",
            ColorLabel = root.TryGetProperty("color_label", out var colorLabel) ? colorLabel.GetString() ?? "unknown" : "unknown",
            AutoClass = root.TryGetProperty("auto_class", out var autoClass) ? autoClass.GetString() ?? "unknown" : "unknown",
            IsWaste = root.TryGetProperty("is_waste", out var isWaste) && isWaste.GetBoolean(),
            WasteReason = root.TryGetProperty("waste_reason", out var wasteReason) ? wasteReason.GetString() ?? string.Empty : string.Empty,
            CompositionScore = root.TryGetProperty("composition_score", out var composition) ? composition.GetSingle() : 0,
            ExposureQuality = root.TryGetProperty("exposure_quality", out var exposureQ) ? exposureQ.GetSingle() : 0,
            RawJson = rawJson
        };

        if (root.TryGetProperty("dominant_colors", out var dominantColors) && dominantColors.ValueKind == JsonValueKind.Array)
        {
            foreach (var color in dominantColors.EnumerateArray())
            {
                var val = color.GetString();
                if (!string.IsNullOrWhiteSpace(val))
                {
                    result.DominantColors.Add(val);
                }
            }
        }

        if (root.TryGetProperty("plugins", out var plugins) && plugins.ValueKind == JsonValueKind.Array)
        {
            foreach (var plugin in plugins.EnumerateArray())
            {
                var pluginResult = new PluginResult
                {
                    PluginName = plugin.TryGetProperty("plugin_name", out var name) ? name.GetString() ?? string.Empty : string.Empty,
                    Score = plugin.TryGetProperty("score", out var score) ? score.GetSingle() : 0
                };

                if (plugin.TryGetProperty("objects", out var objects) && objects.ValueKind == JsonValueKind.Array)
                {
                    foreach (var obj in objects.EnumerateArray())
                    {
                        pluginResult.Objects.Add(new DetectedObject
                        {
                            Label = obj.TryGetProperty("label", out var label) ? label.GetString() ?? string.Empty : string.Empty,
                            Confidence = obj.TryGetProperty("confidence", out var confidence) ? confidence.GetSingle() : 0,
                            X1 = obj.TryGetProperty("x1", out var x1) ? x1.GetSingle() : 0,
                            Y1 = obj.TryGetProperty("y1", out var y1) ? y1.GetSingle() : 0,
                            X2 = obj.TryGetProperty("x2", out var x2) ? x2.GetSingle() : 0,
                            Y2 = obj.TryGetProperty("y2", out var y2) ? y2.GetSingle() : 0,
                        });
                    }
                }

                result.Results.Add(pluginResult);
            }
        }

        return result;
    }
}
