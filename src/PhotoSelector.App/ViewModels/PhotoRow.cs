using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Windows.Media.Imaging;
using PhotoSelector.App.Services;
using PhotoSelector.Domain.Models;

namespace PhotoSelector.App.ViewModels;

public sealed class PhotoRow(PhotoItem photo) : INotifyPropertyChanged
{
    private BitmapSource? _thumbnail;

    public event PropertyChangedEventHandler? PropertyChanged;

    public PhotoItem Photo { get; } = photo;

    public static Func<string, string>? PersonNameResolver { get; set; }

    public string FileName => Photo.FileName;
    public string Path => Photo.Path;
    public float OverallScore => Photo.Analysis.OverallScore;
    public float Sharpness => Photo.Analysis.SharpnessScore;
    public bool EyesClosed => Photo.Analysis.EyesClosed;
    public bool Duplicate => Photo.Analysis.IsDuplicate;
    public bool IsAnalyzed => Photo.Analysis.IsAnalyzed;
    public int FaceCount => Photo.Analysis.FaceCount;
    public string PersonLabel => Photo.Analysis.PersonLabel;
    public string StyleLabel => Photo.Analysis.StyleLabel;
    public string ColorLabel => Photo.Analysis.ColorLabel;
    public string AutoClass => Photo.Analysis.AutoClass;
    public bool IsWaste => Photo.Analysis.IsWaste;
    public string WasteReason => Photo.Analysis.WasteReason;
    public int Rating => Photo.Analysis.Rating;
    public bool IsPicked => Photo.Analysis.IsPicked;
    public BitmapSource? Thumbnail => _thumbnail ??= LoadBitmap(240);
    public string Color1 => GetColorAt(0);
    public string Color2 => GetColorAt(1);
    public string Color3 => GetColorAt(2);
    public string Color4 => GetColorAt(3);
    public string Color5 => GetColorAt(4);

    public string ScoreBadge => IsAnalyzed ? $"{OverallScore:F2}" : "--";
    public string MetaBadge => IsAnalyzed
        ? $"Face:{FaceCount} | {ResolvePersonName()} | {StyleLabel} | {ColorLabel} | {AutoClass}"
        : "未分析";
    public string WasteBadge => IsWaste ? $"废片: {WasteReason}" : string.Empty;

    public void Refresh()
    {
        _thumbnail = null;
        OnPropertyChanged(nameof(OverallScore));
        OnPropertyChanged(nameof(Sharpness));
        OnPropertyChanged(nameof(EyesClosed));
        OnPropertyChanged(nameof(Duplicate));
        OnPropertyChanged(nameof(IsAnalyzed));
        OnPropertyChanged(nameof(FaceCount));
        OnPropertyChanged(nameof(PersonLabel));
        OnPropertyChanged(nameof(StyleLabel));
        OnPropertyChanged(nameof(ColorLabel));
        OnPropertyChanged(nameof(AutoClass));
        OnPropertyChanged(nameof(IsWaste));
        OnPropertyChanged(nameof(WasteReason));
        OnPropertyChanged(nameof(Rating));
        OnPropertyChanged(nameof(IsPicked));
        OnPropertyChanged(nameof(ScoreBadge));
        OnPropertyChanged(nameof(MetaBadge));
        OnPropertyChanged(nameof(WasteBadge));
        OnPropertyChanged(nameof(Color1));
        OnPropertyChanged(nameof(Color2));
        OnPropertyChanged(nameof(Color3));
        OnPropertyChanged(nameof(Color4));
        OnPropertyChanged(nameof(Color5));
        OnPropertyChanged(nameof(Thumbnail));
    }

    private string ResolvePersonName()
    {
        var label = PersonLabel;
        if (PersonNameResolver is not null)
        {
            return PersonNameResolver(label);
        }

        return label;
    }

    private string GetColorAt(int index)
    {
        if (Photo.Analysis.DominantColors is null || Photo.Analysis.DominantColors.Count <= index)
        {
            return string.Empty;
        }

        return Photo.Analysis.DominantColors[index] ?? string.Empty;
    }

    private BitmapSource? LoadBitmap(int decodePixelWidth)
    {
        try
        {
            var sourcePath = string.IsNullOrWhiteSpace(Photo.ThumbnailPath) ? Path : Photo.ThumbnailPath;
            return ImageDisplayService.LoadBitmap(
                sourcePath,
                decodePixelWidth,
                Photo.Analysis.RotationQuarterTurns);
        }
        catch
        {
            return null;
        }
    }

    private void OnPropertyChanged([CallerMemberName] string? name = null)
        => PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
}
