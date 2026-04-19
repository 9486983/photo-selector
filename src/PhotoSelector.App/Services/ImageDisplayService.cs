using System.IO;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace PhotoSelector.App.Services;

public static class ImageDisplayService
{
    private const string ExifOrientationQuery = "/app1/ifd/{ushort=274}";

    public static BitmapSource? LoadBitmap(string path, int? decodePixelWidth = null, int userQuarterTurns = 0)
    {
        if (string.IsNullOrWhiteSpace(path) || !File.Exists(path))
        {
            return null;
        }

        try
        {
            using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite);
            var decoder = BitmapDecoder.Create(
                stream,
                BitmapCreateOptions.PreservePixelFormat,
                BitmapCacheOption.OnLoad);

            if (decoder.Frames.Count == 0)
            {
                return null;
            }

            BitmapSource source = decoder.Frames[0];
            if (decodePixelWidth is > 0)
            {
                var ratio = (double)decodePixelWidth.Value / Math.Max(1, source.PixelWidth);
                if (ratio < 1)
                {
                    var resized = new TransformedBitmap(source, new ScaleTransform(ratio, ratio));
                    resized.Freeze();
                    source = resized;
                }
            }

            var transform = BuildTransform(ReadExifOrientation(decoder.Frames[0].Metadata as BitmapMetadata), userQuarterTurns);
            if (transform is not null)
            {
                var transformed = new TransformedBitmap(source, transform);
                transformed.Freeze();
                return transformed;
            }

            source.Freeze();
            return source;
        }
        catch
        {
            return LoadBitmapFallback(path, decodePixelWidth, userQuarterTurns);
        }
    }

    private static BitmapSource? LoadBitmapFallback(string path, int? decodePixelWidth, int userQuarterTurns)
    {
        try
        {
            var bitmap = new BitmapImage();
            bitmap.BeginInit();
            bitmap.CacheOption = BitmapCacheOption.OnLoad;
            if (decodePixelWidth is > 0)
            {
                bitmap.DecodePixelWidth = decodePixelWidth.Value;
            }

            bitmap.UriSource = new Uri(path, UriKind.Absolute);
            bitmap.EndInit();
            bitmap.Freeze();

            var transform = BuildTransform(1, userQuarterTurns);
            if (transform is null)
            {
                return bitmap;
            }

            var transformed = new TransformedBitmap(bitmap, transform);
            transformed.Freeze();
            return transformed;
        }
        catch
        {
            return null;
        }
    }

    private static int ReadExifOrientation(BitmapMetadata? metadata)
    {
        try
        {
            if (metadata is null || !metadata.ContainsQuery(ExifOrientationQuery))
            {
                return 1;
            }

            var value = metadata.GetQuery(ExifOrientationQuery);
            return value switch
            {
                ushort u => u,
                byte b => b,
                _ => 1
            };
        }
        catch
        {
            return 1;
        }
    }

    private static Transform? BuildTransform(int exifOrientation, int userQuarterTurns)
    {
        var transforms = new TransformGroup();

        switch (exifOrientation)
        {
            case 2:
                transforms.Children.Add(new ScaleTransform(-1, 1));
                break;
            case 3:
                transforms.Children.Add(new RotateTransform(180));
                break;
            case 4:
                transforms.Children.Add(new ScaleTransform(1, -1));
                break;
            case 5:
                transforms.Children.Add(new RotateTransform(90));
                transforms.Children.Add(new ScaleTransform(-1, 1));
                break;
            case 6:
                transforms.Children.Add(new RotateTransform(90));
                break;
            case 7:
                transforms.Children.Add(new RotateTransform(270));
                transforms.Children.Add(new ScaleTransform(-1, 1));
                break;
            case 8:
                transforms.Children.Add(new RotateTransform(270));
                break;
        }

        var normalizedTurns = ((userQuarterTurns % 4) + 4) % 4;
        if (normalizedTurns != 0)
        {
            transforms.Children.Add(new RotateTransform(normalizedTurns * 90));
        }

        return transforms.Children.Count == 0 ? null : transforms;
    }
}
