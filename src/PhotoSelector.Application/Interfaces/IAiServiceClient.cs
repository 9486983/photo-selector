using PhotoSelector.Domain.Models;

namespace PhotoSelector.Application.Interfaces;

public sealed class AnalyzeResponse
{
    public float OverallScore { get; set; }
    public float SharpnessScore { get; set; }
    public float ExposureScore { get; set; }
    public bool EyesClosed { get; set; }
    public bool IsDuplicate { get; set; }
    public int FaceCount { get; set; }
    public string PersonLabel { get; set; } = "none";
    public string StyleLabel { get; set; } = "unknown";
    public string ColorLabel { get; set; } = "unknown";
    public List<string> DominantColors { get; set; } = new();
    public string AutoClass { get; set; } = "unknown";
    public bool IsWaste { get; set; }
    public string WasteReason { get; set; } = string.Empty;
    public float CompositionScore { get; set; }
    public float ExposureQuality { get; set; }
    public List<PluginResult> Results { get; set; } = new();
    public string RawJson { get; set; } = string.Empty;
}

public interface IAiServiceClient
{
    Task<AnalyzeResponse> AnalyzeAsync(string imagePath, CancellationToken cancellationToken = default);
    Task<IReadOnlyCollection<(string ImagePath, AnalyzeResponse Response)>> AnalyzeBatchAsync(
        IReadOnlyCollection<string> imagePaths,
        CancellationToken cancellationToken = default);
}
