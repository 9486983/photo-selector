using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using System.Windows.Media.Imaging;

namespace PhotoSelector.App.Services;

public sealed class ThumbnailCacheService
{
    private const int ExifOrientationId = 0x0112;
    private readonly string _root;

    public ThumbnailCacheService(string root)
    {
        _root = root;
        System.IO.Directory.CreateDirectory(_root);
    }

    public string GetThumbnailPath(string sourcePath, string? libraryFolder = null)
    {
        var folderKey = BuildFolderKey(libraryFolder ?? Path.GetDirectoryName(sourcePath) ?? string.Empty);
        var targetDir = Path.Combine(_root, folderKey);
        System.IO.Directory.CreateDirectory(targetDir);

        var hash = Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(sourcePath))).ToLowerInvariant();
        return Path.Combine(targetDir, $"{hash}.jpg");
    }

    public async Task<string> EnsureThumbnailAsync(
        string sourcePath,
        int maxSide,
        string? libraryFolder = null,
        CancellationToken cancellationToken = default)
    {
        var target = GetThumbnailPath(sourcePath, libraryFolder);
        if (File.Exists(target))
        {
            return target;
        }

        await Task.Run(() =>
        {
            cancellationToken.ThrowIfCancellationRequested();
            try
            {
                using var source = Image.FromFile(sourcePath);
                ApplyExifOrientation(source);

                var ratio = Math.Min((double)maxSide / source.Width, (double)maxSide / source.Height);
                ratio = Math.Min(1.0, Math.Max(0.01, ratio));
                var width = Math.Max(1, (int)Math.Round(source.Width * ratio));
                var height = Math.Max(1, (int)Math.Round(source.Height * ratio));

                using var thumb = new Bitmap(width, height);
                using (var g = Graphics.FromImage(thumb))
                {
                    g.InterpolationMode = InterpolationMode.HighQualityBicubic;
                    g.SmoothingMode = SmoothingMode.HighQuality;
                    g.PixelOffsetMode = PixelOffsetMode.HighQuality;
                    g.DrawImage(source, 0, 0, width, height);
                }

                SaveJpeg(thumb, target);
                return;
            }
            catch (OutOfMemoryException)
            {
                // Fall through to RAW-friendly paths.
            }
            catch
            {
                // Fall through to RAW-friendly paths.
            }

            if (TryCreateThumbnailWithWpf(sourcePath, target, maxSide))
            {
                return;
            }

            CreatePlaceholder(target, maxSide);
        }, cancellationToken);

        return target;
    }

    private static void ApplyExifOrientation(Image image)
    {
        try
        {
            if (!image.PropertyIdList.Contains(ExifOrientationId))
            {
                return;
            }

            var bytes = image.GetPropertyItem(ExifOrientationId)?.Value;
            if (bytes is null || bytes.Length == 0)
            {
                return;
            }

            var orientation = bytes[0];
            var rotateFlip = orientation switch
            {
                2 => RotateFlipType.RotateNoneFlipX,
                3 => RotateFlipType.Rotate180FlipNone,
                4 => RotateFlipType.Rotate180FlipX,
                5 => RotateFlipType.Rotate90FlipX,
                6 => RotateFlipType.Rotate90FlipNone,
                7 => RotateFlipType.Rotate270FlipX,
                8 => RotateFlipType.Rotate270FlipNone,
                _ => RotateFlipType.RotateNoneFlipNone,
            };

            if (rotateFlip != RotateFlipType.RotateNoneFlipNone)
            {
                image.RotateFlip(rotateFlip);
            }
        }
        catch
        {
            // Ignore orientation read failures.
        }
    }

    private static string BuildFolderKey(string folderPath)
    {
        var hash = Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(folderPath))).ToLowerInvariant();
        return hash[..12];
    }

    private static ImageCodecInfo? GetJpegEncoder()
    {
        return ImageCodecInfo.GetImageEncoders().FirstOrDefault(c => c.FormatID == ImageFormat.Jpeg.Guid);
    }

    private static void SaveJpeg(Image image, string target)
    {
        var encoder = GetJpegEncoder();
        if (encoder is null)
        {
            image.Save(target, ImageFormat.Jpeg);
            return;
        }

        using var qualityParams = new EncoderParameters(1);
        qualityParams.Param[0] = new EncoderParameter(System.Drawing.Imaging.Encoder.Quality, 82L);
        image.Save(target, encoder, qualityParams);
    }

    private static bool TryCreateThumbnailWithWpf(string sourcePath, string target, int maxSide)
    {
        try
        {
            var bitmap = ImageDisplayService.LoadBitmap(sourcePath, maxSide);
            if (bitmap is null)
            {
                return false;
            }

            var encoder = new JpegBitmapEncoder();
            encoder.QualityLevel = 82;
            encoder.Frames.Add(BitmapFrame.Create(bitmap));
            using var stream = new FileStream(target, FileMode.Create, FileAccess.Write, FileShare.None);
            encoder.Save(stream);
            return true;
        }
        catch
        {
            return false;
        }
    }

    private static void CreatePlaceholder(string target, int maxSide)
    {
        try
        {
            var size = Math.Max(64, Math.Min(256, maxSide));
            using var bmp = new Bitmap(size, size);
            using var g = Graphics.FromImage(bmp);
            g.Clear(Color.FromArgb(242, 242, 242));
            using var pen = new Pen(Color.FromArgb(200, 210, 227), 2);
            g.DrawRectangle(pen, 6, 6, size - 12, size - 12);
            SaveJpeg(bmp, target);
        }
        catch
        {
            // Ignore placeholder failures.
        }
    }
}
