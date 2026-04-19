using PhotoSelector.App.Services;
using PhotoSelector.App.ViewModels;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media;

namespace PhotoSelector.App;

public partial class PreviewWindow : Window
{
    private readonly IReadOnlyList<PhotoRow> _rows;
    private readonly Action<PhotoRow, int>? _setRating;
    private readonly Action<PhotoRow, int>? _rotatePhoto;
    private bool _isDragging;
    private System.Windows.Point _lastPoint;
    private int _index;

    public PreviewWindow(
        IReadOnlyList<PhotoRow> rows,
        int index,
        Action<PhotoRow, int>? setRating = null,
        Action<PhotoRow, int>? rotatePhoto = null)
    {
        InitializeComponent();
        _rows = rows;
        _index = Math.Clamp(index, 0, Math.Max(0, rows.Count - 1));
        _setRating = setRating;
        _rotatePhoto = rotatePhoto;
        Focus();
        RenderCurrent();
    }

    private PhotoRow? Current => _rows.Count == 0 ? null : _rows[_index];

    private void RenderCurrent()
    {
        var row = Current;
        if (row is null)
        {
            Close();
            return;
        }

        PreviewImage.Source = ImageDisplayService.LoadBitmap(
            row.Path,
            decodePixelWidth: null,
            row.Photo.Analysis.RotationQuarterTurns);

        UpdateColorDots(row.Photo.Analysis.DominantColors);
        UpdateStars(row.Rating);
        PrevButton.IsEnabled = _index > 0;
        NextButton.IsEnabled = _index < _rows.Count - 1;

        ScaleTransform.ScaleX = 1;
        ScaleTransform.ScaleY = 1;
        TranslateTransform.X = 0;
        TranslateTransform.Y = 0;
    }

    private void UpdateColorDots(IReadOnlyList<string> colors)
    {
        var dots = new[] { ColorDot1, ColorDot2, ColorDot3, ColorDot4, ColorDot5 };
        for (var i = 0; i < dots.Length; i++)
        {
            var hex = i < colors.Count ? colors[i] : "#444444";
            dots[i].Fill = TryBrush(hex);
        }
    }

    private static System.Windows.Media.Brush TryBrush(string hex)
    {
        try
        {
            return (SolidColorBrush)new BrushConverter().ConvertFromString(hex)!;
        }
        catch
        {
            return System.Windows.Media.Brushes.Gray;
        }
    }

    private void UpdateStars(int rating)
    {
        var stars = new[] { Star1, Star2, Star3, Star4, Star5 };
        for (var i = 0; i < stars.Length; i++)
        {
            stars[i].Content = rating >= i + 1 ? "★" : "☆";
        }
    }

    private void PrevButton_OnClick(object sender, RoutedEventArgs e)
    {
        if (_index <= 0)
        {
            return;
        }

        _index--;
        RenderCurrent();
    }

    private void NextButton_OnClick(object sender, RoutedEventArgs e)
    {
        if (_index >= _rows.Count - 1)
        {
            return;
        }

        _index++;
        RenderCurrent();
    }

    private void StarButton_OnClick(object sender, RoutedEventArgs e)
    {
        if (Current is null)
        {
            return;
        }

        if (sender is not System.Windows.Controls.Button btn || !int.TryParse(btn.Tag?.ToString(), out var rating))
        {
            return;
        }

        _setRating?.Invoke(Current, rating);
        UpdateStars(rating);
    }

    private void RotateLeftButton_OnClick(object sender, RoutedEventArgs e)
    {
        RotateCurrent(-1);
    }

    private void RotateRightButton_OnClick(object sender, RoutedEventArgs e)
    {
        RotateCurrent(1);
    }

    private void RotateCurrent(int quarterTurns)
    {
        if (Current is null)
        {
            return;
        }

        _rotatePhoto?.Invoke(Current, quarterTurns);
        RenderCurrent();
    }

    private void PreviewImage_OnMouseWheel(object sender, MouseWheelEventArgs e)
    {
        var delta = e.Delta > 0 ? 0.1 : -0.1;
        var next = Math.Clamp(ScaleTransform.ScaleX + delta, 0.3, 8.0);
        ScaleTransform.ScaleX = next;
        ScaleTransform.ScaleY = next;
    }

    private void PreviewImage_OnMouseLeftButtonDown(object sender, MouseButtonEventArgs e)
    {
        _isDragging = true;
        _lastPoint = e.GetPosition(this);
        PreviewImage.CaptureMouse();
        e.Handled = true;
    }

    private void PreviewImage_OnMouseLeftButtonUp(object sender, MouseButtonEventArgs e)
    {
        if (_isDragging)
        {
            _isDragging = false;
            PreviewImage.ReleaseMouseCapture();
            return;
        }

        Close();
    }

    private void PreviewImage_OnMouseMove(object sender, System.Windows.Input.MouseEventArgs e)
    {
        if (!_isDragging)
        {
            return;
        }

        var current = e.GetPosition(this);
        var dx = current.X - _lastPoint.X;
        var dy = current.Y - _lastPoint.Y;
        _lastPoint = current;
        TranslateTransform.X += dx;
        TranslateTransform.Y += dy;
    }

    private void Window_OnMouseLeftButtonDown(object sender, MouseButtonEventArgs e)
    {
        if (e.OriginalSource is DependencyObject obj && IsInsideInteractive(obj))
        {
            return;
        }

        Close();
    }

    private static bool IsInsideInteractive(DependencyObject obj)
    {
        while (obj is not null)
        {
            if (obj is System.Windows.Controls.Button)
            {
                return true;
            }

            obj = VisualTreeHelper.GetParent(obj);
        }

        return false;
    }

    private void Window_OnKeyDown(object sender, System.Windows.Input.KeyEventArgs e)
    {
        if (e.Key == Key.Escape)
        {
            Close();
            return;
        }

        if (e.Key == Key.Left)
        {
            PrevButton_OnClick(this, new RoutedEventArgs());
            return;
        }

        if (e.Key == Key.Right)
        {
            NextButton_OnClick(this, new RoutedEventArgs());
            return;
        }

        if (e.Key == Key.Q)
        {
            RotateCurrent(-1);
            return;
        }

        if (e.Key == Key.E)
        {
            RotateCurrent(1);
            return;
        }

        if (e.Key == Key.Add || e.Key == Key.OemPlus)
        {
            var next = Math.Clamp(ScaleTransform.ScaleX + 0.1, 0.3, 8.0);
            ScaleTransform.ScaleX = next;
            ScaleTransform.ScaleY = next;
            return;
        }

        if (e.Key == Key.Subtract || e.Key == Key.OemMinus)
        {
            var next = Math.Clamp(ScaleTransform.ScaleX - 0.1, 0.3, 8.0);
            ScaleTransform.ScaleX = next;
            ScaleTransform.ScaleY = next;
            return;
        }
    }
}
