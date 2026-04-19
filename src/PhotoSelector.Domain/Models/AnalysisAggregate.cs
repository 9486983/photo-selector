namespace PhotoSelector.Domain.Models;

public sealed class AnalysisAggregate
{
    public bool IsAnalyzed { get; set; }
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
    public int Rating { get; set; }
    public bool IsPicked { get; set; }
    public int RotationQuarterTurns { get; set; }
    public List<PluginResult> PluginResults { get; set; } = new();
    public string RawJson { get; set; } = string.Empty;
}
