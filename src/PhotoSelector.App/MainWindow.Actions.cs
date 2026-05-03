using System.IO;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Controls.Primitives;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using PhotoSelector.App.Services;
using PhotoSelector.App.ViewModels;
using PhotoSelector.Domain.Models;
using PhotoSelector.Domain.Rules;
using Forms = System.Windows.Forms;

namespace PhotoSelector.App;

public partial class MainWindow
{
    private static readonly HashSet<string> RawExtensions = new(StringComparer.OrdinalIgnoreCase)
    {
        ".dng", ".rwa", ".arw", ".cr2", ".cr3", ".nef", ".rw2", ".orf", ".raf", ".sr2", ".srw", ".pef", ".3fr"
    };

    private static readonly HashSet<string> PreferredJpegExtensions = new(StringComparer.OrdinalIgnoreCase)
    {
        ".jpg", ".jpeg"
    };

    private async Task OpenFolderFromRowAsync(LibraryFolderRow row)
    {
        if (row is null)
        {
            return;
        }

        await EnterFolderPageAsync(row.Key);
    }

    private async Task RemoveFolderAsync(LibraryFolderRow row, bool deleteDisk)
    {
        if (row is null)
        {
            return;
        }

        var confirmText = deleteDisk
            ? $"确认删除磁盘文件夹？\n{row.FolderPath}\n该操作会删除文件夹及其中图片。"
            : $"确认从图库移除文件夹？\n{row.FolderPath}\n不会删除磁盘原文件。";
        var confirm = System.Windows.MessageBox.Show(
            confirmText,
            "确认操作",
            MessageBoxButton.YesNo,
            MessageBoxImage.Warning);
        if (confirm != MessageBoxResult.Yes)
        {
            return;
        }

        var token = BeginOperation(deleteDisk ? "删除文件夹中..." : "从图库移除中...");
        try
        {
            var targetRows = _rows
                .Where(r => r.Photo.LibraryFolder.Equals(row.Key, StringComparison.OrdinalIgnoreCase))
                .ToList();

            foreach (var targetRow in targetRows)
            {
                token.ThrowIfCancellationRequested();

                if (!string.IsNullOrWhiteSpace(targetRow.Photo.ThumbnailPath) && File.Exists(targetRow.Photo.ThumbnailPath))
                {
                    try
                    {
                        File.Delete(targetRow.Photo.ThumbnailPath);
                    }
                    catch
                    {
                        // Ignore thumbnail cleanup failures.
                    }
                }

                _rows.Remove(targetRow);
                _photos.Remove(targetRow.Photo);
            }

            _visibleRows.Clear();

            if (deleteDisk && Directory.Exists(row.FolderPath))
            {
                Directory.Delete(row.FolderPath, true);
            }

            if (_selectedLibraryKey?.Equals(row.Key, StringComparison.OrdinalIgnoreCase) == true)
            {
                _selectedLibraryKey = null;
            }

            await SaveStateAsync(token);
            RebuildLibraries();
            EnterGalleryPage();
            StatusText.Text = deleteDisk ? "磁盘文件夹已删除" : "已从图库移除";
        }
        catch (OperationCanceledException)
        {
            StatusText.Text = "操作已取消";
        }
        catch (Exception ex)
        {
            StatusText.Text = $"操作失败: {ex.Message}";
        }
        finally
        {
            EndOperation(StatusText.Text);
        }
    }

    private async void OpenLibraryFolder_OnClick(object sender, RoutedEventArgs e)
    {
        if ((sender as System.Windows.Controls.Button)?.CommandParameter is not LibraryFolderRow row)
        {
            return;
        }

        await OpenFolderFromRowAsync(row);
    }

    private async void OpenLibraryFolderFromMenu_OnClick(object sender, RoutedEventArgs e)
    {
        if ((sender as MenuItem)?.CommandParameter is not LibraryFolderRow row)
        {
            return;
        }

        await OpenFolderFromRowAsync(row);
    }

    private async void RemoveLibraryFolder_OnClick(object sender, RoutedEventArgs e)
    {
        if ((sender as MenuItem)?.CommandParameter is not LibraryFolderRow row)
        {
            return;
        }

        await RemoveFolderAsync(row, deleteDisk: false);
    }

    private async void DeleteLibraryFolderDisk_OnClick(object sender, RoutedEventArgs e)
    {
        if ((sender as MenuItem)?.CommandParameter is not LibraryFolderRow row)
        {
            return;
        }

        await RemoveFolderAsync(row, deleteDisk: true);
    }

    private async void ImportButton_OnClick(object sender, RoutedEventArgs e)
    {
        using var dialog = new Forms.FolderBrowserDialog();
        if (dialog.ShowDialog() != Forms.DialogResult.OK || string.IsNullOrWhiteSpace(dialog.SelectedPath))
        {
            return;
        }

        await ImportFolderAsync(dialog.SelectedPath);
    }

    private async void GalleryPage_OnDrop(object sender, System.Windows.DragEventArgs e)
    {
        if (!e.Data.GetDataPresent(System.Windows.DataFormats.FileDrop))
        {
            return;
        }

        var dropped = (string[]?)e.Data.GetData(System.Windows.DataFormats.FileDrop) ?? Array.Empty<string>();
        var folders = dropped.Where(Directory.Exists).Distinct(StringComparer.OrdinalIgnoreCase).ToList();
        if (folders.Count() == 0)
        {
            return;
        }

        foreach (var folder in folders)
        {
            await ImportFolderAsync(folder);
        }
    }

    private void GalleryPage_OnDragOver(object sender, System.Windows.DragEventArgs e)
    {
        e.Effects = e.Data.GetDataPresent(System.Windows.DataFormats.FileDrop)
            ? System.Windows.DragDropEffects.Copy
            : System.Windows.DragDropEffects.None;
        e.Handled = true;
    }

    private async Task ImportFolderAsync(string folder)
    {
        var token = BeginOperation($"导入 {Path.GetFileName(folder)} ...");
        try
        {
            var targetFolder = PrepareImportFolder(folder, token);
            var imported = await Task.Run(() => _importService.ImportFromDirectory(targetFolder, recursive: true), token);
            var newItems = new List<PhotoItem>();
            foreach (var item in imported)
            {
                token.ThrowIfCancellationRequested();
                if (_photos.Any(p => p.Path.Equals(item.Path, StringComparison.OrdinalIgnoreCase)))
                {
                    continue;
                }

                item.LibraryFolder = Path.GetFullPath(targetFolder);
                _photos.Add(item);
                _rows.Add(new PhotoRow(item));
                newItems.Add(item);
            }

            await EnsureThumbnailsAsync(newItems, "生成缩略图中", token);
            await SaveStateAsync(token);
            RebuildLibraries();
            StatusText.Text = $"导入完成: {Path.GetFileName(targetFolder)}，新增 {newItems.Count} 张";
        }
        catch (OperationCanceledException)
        {
            StatusText.Text = "导入已取消";
        }
        finally
        {
            EndOperation(StatusText.Text);
        }
    }

    private string PrepareImportFolder(string folder, CancellationToken token)
    {
        SortFilesByExtension(folder, token);

        var jpgFolder = FindPreferredFolder(folder, PreferredJpegExtensions);
        if (!string.IsNullOrWhiteSpace(jpgFolder))
        {
            return jpgFolder;
        }

        var rawFolder = FindPreferredFolder(folder, RawExtensions);
        return string.IsNullOrWhiteSpace(rawFolder) ? folder : rawFolder;
    }

    private static void SortFilesByExtension(string folder, CancellationToken token)
    {
        if (!Directory.Exists(folder))
        {
            return;
        }

        foreach (var file in Directory.EnumerateFiles(folder, "*", SearchOption.TopDirectoryOnly))
        {
            token.ThrowIfCancellationRequested();

            var extension = Path.GetExtension(file);
            if (string.IsNullOrWhiteSpace(extension))
            {
                continue;
            }

            var normalized = extension.TrimStart('.').ToUpperInvariant();
            if (string.IsNullOrWhiteSpace(normalized))
            {
                continue;
            }

            var targetDir = Path.Combine(folder, normalized);
            Directory.CreateDirectory(targetDir);

            var destPath = Path.Combine(targetDir, Path.GetFileName(file));
            destPath = GetAvailablePath(destPath);
            File.Move(file, destPath);
        }
    }

    private static string? FindPreferredFolder(string folder, IEnumerable<string> extensions)
    {
        foreach (var ext in extensions)
        {
            var name = ext.TrimStart('.').ToUpperInvariant();
            var candidate = Path.Combine(folder, name);
            if (!Directory.Exists(candidate))
            {
                continue;
            }

            if (Directory.EnumerateFiles(candidate, "*", SearchOption.TopDirectoryOnly).Any())
            {
                return candidate;
            }
        }

        return null;
    }

    private static string GetAvailablePath(string path)
    {
        if (!File.Exists(path))
        {
            return path;
        }

        var directory = Path.GetDirectoryName(path) ?? string.Empty;
        var name = Path.GetFileNameWithoutExtension(path);
        var extension = Path.GetExtension(path);
        var index = 1;

        while (true)
        {
            var candidate = Path.Combine(directory, $"{name}_{index}{extension}");
            if (!File.Exists(candidate))
            {
                return candidate;
            }

            index++;
        }
    }

    private async Task EnsureThumbnailsAsync(IReadOnlyCollection<PhotoItem> photos, string stepMessage, CancellationToken token)
    {
        if (photos.Count == 0)
        {
            return;
        }

        var list = photos.ToList();
        var maxParallel = Math.Clamp(Environment.ProcessorCount / 2, 2, 6);
        using var gate = new SemaphoreSlim(maxParallel);
        var processed = 0;

        var tasks = list.Select(async photo =>
        {
            await gate.WaitAsync(token);
            try
            {
                token.ThrowIfCancellationRequested();

                var current = Interlocked.Increment(ref processed);
                await Dispatcher.InvokeAsync(() =>
                {
                    OperationText.Text = $"{stepMessage} {current}/{list.Count}";
                });

                if (!File.Exists(photo.Path))
                {
                    return;
                }

                if (string.IsNullOrWhiteSpace(photo.ThumbnailPath) || !File.Exists(photo.ThumbnailPath))
                {
                    photo.ThumbnailPath = await _thumbnailCache.EnsureThumbnailAsync(
                        photo.Path,
                        960,
                        photo.LibraryFolder,
                        token);
                    var row = _rows.FirstOrDefault(r => r.Photo.Id == photo.Id);
                    row?.Refresh();
                }
            }
            finally
            {
                gate.Release();
            }
        });

        await Task.WhenAll(tasks);
    }

    private async void AnalyzeSelectedButton_OnClick(object sender, RoutedEventArgs e)
    {
        if (PhotoList.SelectedItem is not PhotoRow row)
        {
            StatusText.Text = "请先选择一张照片";
            return;
        }

        await AnalyzeRowsAsync(new[] { row });
    }

    private async void AnalyzeAllButton_OnClick(object sender, RoutedEventArgs e)
    {
        if (_visibleRows.Count == 0)
        {
            StatusText.Text = "当前筛选下没有可分析照片";
            return;
        }

        await AnalyzeRowsAsync(_visibleRows.ToList());
    }

    private async Task AnalyzeRowsAsync(IReadOnlyCollection<PhotoRow> rows)
    {
        var token = BeginOperation("AI分析中...");
        try
        {
            var photos = rows.Select(r => r.Photo).ToList();
            await EnsureThumbnailsAsync(photos, "准备缩略图", token);

            const int batchSize = 12;
            for (var start = 0; start < rows.Count; start += batchSize)
            {
                token.ThrowIfCancellationRequested();
                var batch = rows.Skip(start).Take(batchSize).ToList();
                OperationText.Text = $"分析中 {Math.Min(start + batch.Count, rows.Count)}/{rows.Count}";

                await _analysisCoordinator.AnalyzeBatchAsync(
                    batch.Select(r => r.Photo).ToList(),
                    p => string.IsNullOrWhiteSpace(p.ThumbnailPath) ? p.Path : p.ThumbnailPath,
                    token);

                foreach (var row in batch)
                {
                    row.Photo.Analysis.StyleLabel = CanonicalGroup("style", row.Photo.Analysis.StyleLabel);
                    row.Photo.Analysis.ColorLabel = CanonicalGroup("color", row.Photo.Analysis.ColorLabel);
                    row.Photo.Analysis.AutoClass = CanonicalClass(row.Photo.Analysis.AutoClass);
                    row.Refresh();
                }
            }

            await SaveStateAsync(token);
            RebuildLibraries();
            RebuildGroups();
            ApplyFilters();
            StatusText.Text = $"分析完成，共 {rows.Count} 张";
        }
        catch (OperationCanceledException)
        {
            StatusText.Text = "分析已取消";
        }
        catch (Exception ex)
        {
            StatusText.Text = $"分析失败: {ex.Message}";
        }
        finally
        {
            EndOperation(StatusText.Text);
        }
    }

    private void PhotoList_OnSelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        if (PhotoList.SelectedItem is not PhotoRow row)
        {
            PreviewImage.Source = null;
            DetailOverallText.Text = string.Empty;
            DetailSharpnessText.Text = string.Empty;
            DetailExposureText.Text = string.Empty;
            DetailFaceText.Text = string.Empty;
            DetailClassText.Text = string.Empty;
            DetailWasteText.Text = string.Empty;
            ExifCard.Visibility = Visibility.Collapsed;
            ExifIsoText.Text = string.Empty;
            ExifApertureText.Text = string.Empty;
            ExifShutterText.Text = string.Empty;
            ExifFocalText.Text = string.Empty;
            ExifCameraText.Text = string.Empty;
            ExifCapturedText.Text = string.Empty;
            ExifCapturedText.Visibility = Visibility.Collapsed;
            DetailRawJsonText.Text = string.Empty;
            return;
        }

        try
        {
            PreviewImage.Source =
                ImageDisplayService.LoadBitmap(row.Path, 960, row.Photo.Analysis.RotationQuarterTurns)
                ?? ImageDisplayService.LoadBitmap(row.Photo.ThumbnailPath, 960, row.Photo.Analysis.RotationQuarterTurns);
        }
        catch
        {
            PreviewImage.Source = null;
        }

        DetailOverallText.Text = $"总分: {row.OverallScore:F3}";
        DetailSharpnessText.Text = $"清晰度: {row.Sharpness:F3}";
        DetailExposureText.Text = $"曝光: {row.Photo.Analysis.ExposureScore:F3}";
        DetailFaceText.Text = $"人像计数: {row.FaceCount} / 人物: {GetPersonDisplayName(row.PersonLabel)}";
        var deviceLabel = string.IsNullOrWhiteSpace(row.Photo.Metadata.CameraModel) && string.IsNullOrWhiteSpace(row.Photo.Metadata.CameraMake)
            ? string.Empty
            : $" / 设备: {string.Join(' ', new[] { row.Photo.Metadata.CameraMake, row.Photo.Metadata.CameraModel }.Where(s => !string.IsNullOrWhiteSpace(s)).Select(s => s!.Trim()))}";
        DetailClassText.Text = $"分类: {row.AutoClass} / 风格: {row.StyleLabel} / 颜色: {row.ColorLabel}{deviceLabel}";
        DetailWasteText.Text = row.IsWaste ? $"废片: {row.WasteReason}" : "废片: 否";

        var meta = row.Photo.Metadata;
        var hasExif = meta.CapturedAt.HasValue
                      || meta.Iso.HasValue
                      || !string.IsNullOrWhiteSpace(meta.Aperture)
                      || !string.IsNullOrWhiteSpace(meta.ShutterSpeed)
                      || !string.IsNullOrWhiteSpace(meta.FocalLength)
                      || !string.IsNullOrWhiteSpace(meta.CameraMake)
                      || !string.IsNullOrWhiteSpace(meta.CameraModel)
                      || !string.IsNullOrWhiteSpace(meta.LensModel);
        if (hasExif)
        {
            ExifCard.Visibility = Visibility.Visible;
            ExifIsoText.Text = meta.Iso.HasValue ? $"ISO: {meta.Iso.Value}" : string.Empty;
            ExifIsoText.Visibility = meta.Iso.HasValue ? Visibility.Visible : Visibility.Collapsed;

            ExifApertureText.Text = !string.IsNullOrWhiteSpace(meta.Aperture) ? $"光圈: {meta.Aperture}" : string.Empty;
            ExifApertureText.Visibility = !string.IsNullOrWhiteSpace(meta.Aperture) ? Visibility.Visible : Visibility.Collapsed;

            ExifShutterText.Text = !string.IsNullOrWhiteSpace(meta.ShutterSpeed) ? $"快门: {meta.ShutterSpeed}" : string.Empty;
            ExifShutterText.Visibility = !string.IsNullOrWhiteSpace(meta.ShutterSpeed) ? Visibility.Visible : Visibility.Collapsed;

            ExifFocalText.Text = !string.IsNullOrWhiteSpace(meta.FocalLength) ? $"焦距: {meta.FocalLength}" : string.Empty;
            ExifFocalText.Visibility = !string.IsNullOrWhiteSpace(meta.FocalLength) ? Visibility.Visible : Visibility.Collapsed;

            var cameraParts = new List<string>();
            if (!string.IsNullOrWhiteSpace(meta.CameraMake))
            {
                cameraParts.Add(meta.CameraMake);
            }
            if (!string.IsNullOrWhiteSpace(meta.CameraModel))
            {
                cameraParts.Add(meta.CameraModel);
            }
            if (!string.IsNullOrWhiteSpace(meta.LensModel))
            {
                cameraParts.Add(meta.LensModel);
            }

            var cameraText = cameraParts.Count > 0 ? $"设备: {string.Join(" / ", cameraParts)}" : string.Empty;
            ExifCameraText.Text = cameraText;
            ExifCameraText.Visibility = cameraParts.Count > 0 ? Visibility.Visible : Visibility.Collapsed;

            var captured = meta.CapturedAt ?? row.Photo.ImportedAt;
            ExifCapturedText.Text = $"拍摄时间: {captured:yyyy-MM-dd HH:mm:ss}";
            ExifCapturedText.Visibility = Visibility.Visible;
        }
        else
        {
            ExifCard.Visibility = Visibility.Collapsed;
        }
        DetailRawJsonText.Text = string.IsNullOrWhiteSpace(row.Photo.Analysis.RawJson)
            ? JsonSerializer.Serialize(row.Photo, new JsonSerializerOptions { WriteIndented = true })
            : row.Photo.Analysis.RawJson;
    }

    private void ApplyRuleButton_OnClick(object sender, RoutedEventArgs e)
    {
        var expression = RuleTextBox.Text;
        if (string.IsNullOrWhiteSpace(expression))
        {
            StatusText.Text = "规则不能为空";
            return;
        }

        var scoped = GetLibraryScopedRows().ToList();
        var passing = new List<PhotoRow>();
        foreach (var row in scoped)
        {
            var result = _ruleEngine.Evaluate(expression, new RuleContext { Photo = row.Photo }, "QuickRule");
            if (result.Passed)
            {
                passing.Add(row);
            }
        }

        _visibleRows.Clear();
        foreach (var row in passing)
        {
            _visibleRows.Add(row);
        }

        StatusText.Text = $"规则通过: {passing.Count}/{scoped.Count}";
    }

    private async void ExportButton_OnClick(object sender, RoutedEventArgs e)
    {
        var dialog = new Microsoft.Win32.SaveFileDialog
        {
            Title = "导出筛选结果",
            Filter = "JSON|*.json",
            FileName = $"selection_{DateTime.Now:yyyyMMdd_HHmmss}.json"
        };

        if (dialog.ShowDialog() != true)
        {
            return;
        }

        var payload = _visibleRows
            .Where(p => p.IsAnalyzed)
            .OrderByDescending(p => p.OverallScore)
            .Select(row => new
            {
                row.Photo.Id,
                row.Photo.Path,
                row.Photo.ThumbnailPath,
                row.Photo.FileName,
                row.Photo.LibraryFolder,
                row.Photo.Analysis.OverallScore,
                row.Photo.Analysis.SharpnessScore,
                row.Photo.Analysis.ExposureScore,
                row.Photo.Analysis.FaceCount,
                row.Photo.Analysis.StyleLabel,
                row.Photo.Analysis.IsWaste,
                row.Photo.Analysis.WasteReason,
                row.Photo.Analysis.EyesClosed,
                row.Photo.Analysis.IsDuplicate
            })
            .ToList();

        await File.WriteAllTextAsync(
            dialog.FileName,
            JsonSerializer.Serialize(payload, new JsonSerializerOptions { WriteIndented = true }),
            Encoding.UTF8);

        StatusText.Text = $"已导出: {dialog.FileName}";
    }

    private async void ManualTagGood_OnClick(object sender, RoutedEventArgs e)
    {
        if (GetRowFromMenu(sender) is not { } row) return;
        row.Photo.Analysis.IsWaste = false;
        row.Photo.Analysis.WasteReason = "manual_good";
        row.Refresh();
        await SubmitFeedbackAsync(row, "good", "manual tag good");
        await SaveStateAsync();
        RebuildGroups();
        ApplyFilters();
    }

    private async void ManualTagWaste_OnClick(object sender, RoutedEventArgs e)
    {
        if (GetRowFromMenu(sender) is not { } row) return;
        row.Photo.Analysis.IsWaste = true;
        row.Photo.Analysis.WasteReason = "manual_waste";
        row.Refresh();
        await SubmitFeedbackAsync(row, "waste", "manual tag waste");
        await SaveStateAsync();
        RebuildGroups();
        ApplyFilters();
    }

    private async void ManualTagPortrait_OnClick(object sender, RoutedEventArgs e)
    {
        if (GetRowFromMenu(sender) is not { } row) return;
        row.Photo.Analysis.FaceCount = Math.Max(1, row.Photo.Analysis.FaceCount);
        row.Refresh();
        await SubmitFeedbackAsync(row, "portrait", "manual tag portrait");
        await SaveStateAsync();
        RebuildGroups();
        ApplyFilters();
    }

    private async void SetRating_OnClick(object sender, RoutedEventArgs e)
    {
        if (sender is not System.Windows.Controls.Button button
            || button.CommandParameter is not PhotoRow row
            || button.Tag is not string tag
            || !int.TryParse(tag, out var rating))
        {
            return;
        }

        row.Photo.Analysis.Rating = Math.Clamp(rating, 1, 5);
        row.Refresh();
        await SaveStateAsync();
        RebuildGroups();
        ApplyFilters();
    }

    private async Task RotatePhotoAsync(PhotoRow row, int quarterTurnDelta)
    {
        row.Photo.Analysis.RotationQuarterTurns = NormalizeQuarterTurns(
            row.Photo.Analysis.RotationQuarterTurns + quarterTurnDelta);
        row.Refresh();

        if (ReferenceEquals(PhotoList.SelectedItem, row))
        {
            PhotoList_OnSelectionChanged(PhotoList, new SelectionChangedEventArgs(
                Selector.SelectionChangedEvent,
                Array.Empty<object>(),
                new[] { row }));
        }

        await SaveStateAsync();
    }

    private async Task AssignPersonAsync(PhotoRow row, string newName)
    {
        var label = row.PersonLabel ?? string.Empty;
        if (!string.IsNullOrWhiteSpace(label) && label.StartsWith("person_", StringComparison.OrdinalIgnoreCase))
        {
            _personNames[label] = newName;
            await SubmitPersonRenameAsync(label, newName);
        }

        row.Photo.Analysis.PersonLabel = newName;
        row.Refresh();
        await SavePersonNamesAsync();
        await SubmitPersonAssignAsync(row.Path, newName);
        await SaveStateAsync();
        RebuildGroups();
        ApplyFilters();
    }

    private async Task ClearPersonAsync(PhotoRow row)
    {
        row.Photo.Analysis.PersonLabel = "none";
        row.Refresh();
        await SaveStateAsync();
        RebuildGroups();
        ApplyFilters();
    }

    private async void Reanalyze_OnClick(object sender, RoutedEventArgs e)
    {
        if (GetRowFromMenu(sender) is not { } row) return;
        await AnalyzeRowsAsync(new[] { row });
    }

    private void OpenRowMenu_OnClick(object sender, RoutedEventArgs e)
    {
        if (sender is not System.Windows.Controls.Button button || button.ContextMenu is not ContextMenu menu)
        {
            return;
        }

        if (button.CommandParameter is PhotoRow row)
        {
            PopulateRowMenu(menu, row);
        }

        menu.Placement = PlacementMode.Bottom;
        menu.PlacementTarget = button;
        menu.IsOpen = true;
    }

    private async void PhotoListItem_OnMouseDoubleClick(object sender, MouseButtonEventArgs e)
    {
        if (sender is not ListBoxItem item || item.DataContext is not PhotoRow row)
        {
            return;
        }

        row.Photo.Analysis.IsPicked = !row.Photo.Analysis.IsPicked;
        row.Refresh();
        await SaveStateAsync();
    }

    private void ToggleDetailsButton_OnClick(object sender, RoutedEventArgs e)
    {
        //if (sender is not System.Windows.Controls.Primitives.ToggleButton toggle)
        //{
        //    return;
        //}

        //var show = toggle.IsChecked != false;
        //DetailsPanel.Visibility = show ? Visibility.Visible : Visibility.Collapsed;
        //DetailsColumn.Width = show ? GridLength.Auto : new GridLength(0);
        //toggle.Content = show ? "收起详情" : "展开详情";
    }

    private async void FinishSelectionButton_OnClick(object sender, RoutedEventArgs e)
    {
        if (string.IsNullOrWhiteSpace(_selectedLibraryKey) || !Directory.Exists(_selectedLibraryKey))
        {
            StatusText.Text = "当前没有可导出的文件夹";
            return;
        }

        var baseFolder = ResolveFinishBaseFolder(_selectedLibraryKey);
        var finishFolder = Path.Combine(baseFolder, "FINISH");
        var jpgFolder = Path.Combine(finishFolder, "JPG");
        var rawFolder = Path.Combine(finishFolder, "RAW");

        Directory.CreateDirectory(jpgFolder);
        Directory.CreateDirectory(rawFolder);

        var selected = GetLibraryScopedRows().Where(r => r.Photo.Analysis.IsPicked).ToList();
        if (selected.Count == 0)
        {
            StatusText.Text = "没有已选中的照片";
            return;
        }

        var token = BeginOperation($"复制已选照片 ({selected.Count})");
        try
        {
            foreach (var row in selected)
            {
                token.ThrowIfCancellationRequested();
                var ext = Path.GetExtension(row.Path);
                var targetRoot = RawExtensions.Contains(ext) ? rawFolder : jpgFolder;
                var dest = Path.Combine(targetRoot, Path.GetFileName(row.Path));
                if (File.Exists(dest))
                {
                    continue;
                }
                dest = GetAvailablePath(dest);
                File.Copy(row.Path, dest, overwrite: false);
            }

            StatusText.Text = $"已完成选片: {selected.Count} 张";
        }
        catch (OperationCanceledException)
        {
            StatusText.Text = "已取消选片导出";
        }
        catch (Exception ex)
        {
            StatusText.Text = $"选片导出失败: {ex.Message}";
        }
        finally
        {
            EndOperation(StatusText.Text);
        }
    }

    private static string ResolveFinishBaseFolder(string libraryFolder)
    {
        var folderName = new DirectoryInfo(libraryFolder).Name;
        if (folderName.Equals("JPG", StringComparison.OrdinalIgnoreCase)
            || folderName.Equals("JPEG", StringComparison.OrdinalIgnoreCase)
            || RawExtensions.Contains("." + folderName))
        {
            return Directory.GetParent(libraryFolder)?.FullName ?? libraryFolder;
        }

        return libraryFolder;
    }

    private void ThumbFooter_OnSizeChanged(object sender, SizeChangedEventArgs e)
    {
        if (sender is Grid grid)
        {
            UpdateThumbFooterLayout(grid);
        }
    }

    private void ThumbFooter_OnLoaded(object sender, RoutedEventArgs e)
    {
        if (sender is not Grid grid)
        {
            return;
        }

        Dispatcher.BeginInvoke(() => UpdateThumbFooterLayout(grid), System.Windows.Threading.DispatcherPriority.Loaded);
    }

    private static void UpdateThumbFooterLayout(Grid grid)
    {
        if (grid.FindName("StarPanel") is not FrameworkElement starPanel
            || grid.FindName("SwatchPanel") is not FrameworkElement swatchPanel)
        {
            return;
        }

        starPanel.Measure(new System.Windows.Size(double.PositiveInfinity, double.PositiveInfinity));
        swatchPanel.Measure(new System.Windows.Size(double.PositiveInfinity, double.PositiveInfinity));

        var required = starPanel.DesiredSize.Width + swatchPanel.DesiredSize.Width + 8;
        var available = grid.ActualWidth;
        if (available <= 0)
        {
            var listBoxItem = FindAncestor<ListBoxItem>(grid);
            if (listBoxItem is not null)
            {
                available = Math.Max(0, listBoxItem.ActualWidth - 16);
            }
        }
        var wrap = available > 0 && required > available;

        if (wrap)
        {
            Grid.SetRow(swatchPanel, 1);
            Grid.SetColumn(swatchPanel, 0);
            Grid.SetColumnSpan(swatchPanel, 2);
            swatchPanel.HorizontalAlignment = System.Windows.HorizontalAlignment.Left;
            swatchPanel.Margin = new Thickness(0, 2, 0, 0);
        }
        else
        {
            Grid.SetRow(swatchPanel, 0);
            Grid.SetColumn(swatchPanel, 1);
            Grid.SetColumnSpan(swatchPanel, 1);
            swatchPanel.HorizontalAlignment = System.Windows.HorizontalAlignment.Right;
            swatchPanel.Margin = new Thickness(0, 2, 0, 0);
        }
    }

    private static T? FindAncestor<T>(DependencyObject? current) where T : DependencyObject
    {
        while (current is not null)
        {
            if (current is T match)
            {
                return match;
            }

            current = VisualTreeHelper.GetParent(current);
        }

        return null;
    }

    private void PhotoListItem_OnPreviewMouseRightButtonDown(object sender, MouseButtonEventArgs e)
    {
        if (sender is not ListBoxItem item || item.DataContext is not PhotoRow row)
        {
            return;
        }

        item.IsSelected = true;

        var menu = new ContextMenu();
        PopulateRowMenu(menu, row);
        menu.Placement = PlacementMode.MousePoint;
        menu.PlacementTarget = item;
        menu.IsOpen = true;
        e.Handled = true;
    }

    private void PopulateRowMenu(ContextMenu menu, PhotoRow row)
    {
        menu.Items.Clear();
        BuildPersonMenu(menu, row);
        menu.Items.Add(new Separator());

        var goodItem = new MenuItem { Header = "手动标记: 好片", CommandParameter = row };
        goodItem.Click += ManualTagGood_OnClick;
        menu.Items.Add(goodItem);

        var wasteItem = new MenuItem { Header = "手动标记: 废片", CommandParameter = row };
        wasteItem.Click += ManualTagWaste_OnClick;
        menu.Items.Add(wasteItem);

        var portraitItem = new MenuItem { Header = "手动标记: 人像", CommandParameter = row };
        portraitItem.Click += ManualTagPortrait_OnClick;
        menu.Items.Add(portraitItem);

        menu.Items.Add(new Separator());
        var rotateLeft = new MenuItem { Header = "左转 90°", CommandParameter = row };
        rotateLeft.Click += RotateLeft_OnClick;
        menu.Items.Add(rotateLeft);

        var rotateRight = new MenuItem { Header = "右转 90°", CommandParameter = row };
        rotateRight.Click += RotateRight_OnClick;
        menu.Items.Add(rotateRight);

        menu.Items.Add(new Separator());
        var reanalyze = new MenuItem { Header = "重新分析", CommandParameter = row };
        reanalyze.Click += Reanalyze_OnClick;
        menu.Items.Add(reanalyze);
    }

    private async void RotateLeft_OnClick(object sender, RoutedEventArgs e)
    {
        if (GetRowFromMenu(sender) is not { } row)
        {
            return;
        }

        await RotatePhotoAsync(row, -1);
    }

    private async void RotateRight_OnClick(object sender, RoutedEventArgs e)
    {
        if (GetRowFromMenu(sender) is not { } row)
        {
            return;
        }

        await RotatePhotoAsync(row, 1);
    }

    private void BuildPersonMenu(ContextMenu menu, PhotoRow row)
    {
        var personMenu = new MenuItem { Header = "人物标记" };
        var hasPerson = !string.IsNullOrWhiteSpace(row.PersonLabel)
                        && !row.PersonLabel.Equals("none", StringComparison.OrdinalIgnoreCase);
        if (hasPerson)
        {
            var clear = new MenuItem { Header = "清除人物标记" };
            clear.Click += async (_, _) => { await ClearPersonAsync(row); };
            personMenu.Items.Add(clear);
        }
        else
        {
            var names = GetKnownPersonNames();
            foreach (var name in names)
            {
                var item = new MenuItem { Header = name, Tag = name };
                item.Click += async (_, _) =>
                {
                    await AssignPersonAsync(row, name);
                };
                personMenu.Items.Add(item);
            }

            var addNew = new MenuItem { Header = "新建..." };
            addNew.Click += async (_, _) =>
            {
                var current = GetPersonDisplayName(row.PersonLabel);
                var input = Microsoft.VisualBasic.Interaction.InputBox("请输入人物名称", "标记人物", current);
                if (string.IsNullOrWhiteSpace(input))
                {
                    return;
                }

                await AssignPersonAsync(row, input.Trim());
            };
            personMenu.Items.Add(addNew);
        }

        menu.Items.Add(personMenu);
    }

    private List<string> GetKnownPersonNames()
    {
        var names = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var value in _personNames.Values)
        {
            if (!string.IsNullOrWhiteSpace(value))
            {
                names.Add(value);
            }
        }

        foreach (var row in _rows)
        {
            if (string.IsNullOrWhiteSpace(row.PersonLabel) || row.PersonLabel.Equals("none", StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }

            names.Add(GetPersonDisplayName(row.PersonLabel));
        }

        return names.OrderBy(n => n, StringComparer.OrdinalIgnoreCase).ToList();
    }

    private void OpenLibraryMenu_OnClick(object sender, RoutedEventArgs e)
    {
        if (sender is not System.Windows.Controls.Button button || button.ContextMenu is not ContextMenu menu)
        {
            return;
        }

        menu.Placement = PlacementMode.Bottom;
        menu.PlacementTarget = button;
        menu.IsOpen = true;
    }

    private void FullscreenPreview_OnClick(object sender, RoutedEventArgs e)
    {
        if ((sender as System.Windows.Controls.Button)?.CommandParameter is not PhotoRow row)
        {
            return;
        }

        var currentRows = _visibleRows.ToList();
        var startIndex = currentRows.FindIndex(r => r.Photo.Id == row.Photo.Id);
        if (startIndex < 0)
        {
            startIndex = 0;
        }

        var preview = new PreviewWindow(currentRows, startIndex, (target, rating) =>
        {
            target.Photo.Analysis.Rating = Math.Clamp(rating, 1, 5);
            target.Refresh();
            RebuildGroups();
            ApplyFilters();
            _ = SaveStateAsync();
        }, (target, quarterTurnDelta) =>
        {
            _ = RotatePhotoAsync(target, quarterTurnDelta);
        })
        { Owner = this };
        preview.ShowDialog();
    }

    private static int NormalizeQuarterTurns(int turns)
    {
        var normalized = turns % 4;
        return normalized < 0 ? normalized + 4 : normalized;
    }
}
