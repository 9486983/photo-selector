using System.Text.Json;
using System.Data.Common;
using Microsoft.Data.Sqlite;
using PhotoSelector.Application.Interfaces;
using PhotoSelector.Domain.Models;

namespace PhotoSelector.Infrastructure.Persistence;

public sealed class SqlitePhotoRepository : IPhotoRepository
{
    private readonly string _dbPath;

    public SqlitePhotoRepository(string dbPath)
    {
        _dbPath = dbPath;
        var dir = Path.GetDirectoryName(_dbPath);
        if (!string.IsNullOrWhiteSpace(dir))
        {
            Directory.CreateDirectory(dir);
        }

        using var connection = new SqliteConnection($"Data Source={_dbPath}");
        connection.Open();
        EnsureSchema(connection);
    }

    public async Task SaveAsync(IReadOnlyCollection<PhotoItem> photos, CancellationToken cancellationToken = default)
    {
        await using var connection = new SqliteConnection($"Data Source={_dbPath}");
        await connection.OpenAsync(cancellationToken);

        await using var tx = await connection.BeginTransactionAsync(cancellationToken);
        await SyncRemovedRowsAsync(connection, tx, photos, cancellationToken);
        foreach (var photo in photos)
        {
            await UpsertPhotoAsync(connection, tx, photo, cancellationToken);
            await UpsertAnalysisAsync(connection, tx, photo, cancellationToken);
        }

        await tx.CommitAsync(cancellationToken);
    }

    private static async Task SyncRemovedRowsAsync(
        SqliteConnection connection,
        DbTransaction tx,
        IReadOnlyCollection<PhotoItem> photos,
        CancellationToken cancellationToken)
    {
        var keepIds = photos.Select(p => p.Id.ToString()).ToHashSet(StringComparer.OrdinalIgnoreCase);
        var existingCmd = connection.CreateCommand();
        existingCmd.Transaction = (SqliteTransaction)tx;
        existingCmd.CommandText = "SELECT Id FROM Image;";

        var staleIds = new List<string>();
        await using (var reader = await existingCmd.ExecuteReaderAsync(cancellationToken))
        {
            while (await reader.ReadAsync(cancellationToken))
            {
                var id = reader.GetString(0);
                if (!keepIds.Contains(id))
                {
                    staleIds.Add(id);
                }
            }
        }

        foreach (var staleId in staleIds)
        {
            var deletePlugin = connection.CreateCommand();
            deletePlugin.Transaction = (SqliteTransaction)tx;
            deletePlugin.CommandText = "DELETE FROM PluginResult WHERE ImageId = $id;";
            deletePlugin.Parameters.AddWithValue("$id", staleId);
            await deletePlugin.ExecuteNonQueryAsync(cancellationToken);

            var deleteAnalysis = connection.CreateCommand();
            deleteAnalysis.Transaction = (SqliteTransaction)tx;
            deleteAnalysis.CommandText = "DELETE FROM Analysis WHERE ImageId = $id;";
            deleteAnalysis.Parameters.AddWithValue("$id", staleId);
            await deleteAnalysis.ExecuteNonQueryAsync(cancellationToken);

            var deleteImage = connection.CreateCommand();
            deleteImage.Transaction = (SqliteTransaction)tx;
            deleteImage.CommandText = "DELETE FROM Image WHERE Id = $id;";
            deleteImage.Parameters.AddWithValue("$id", staleId);
            await deleteImage.ExecuteNonQueryAsync(cancellationToken);
        }
    }

    public async Task<IReadOnlyCollection<PhotoItem>> LoadAsync(CancellationToken cancellationToken = default)
    {
        var result = new List<PhotoItem>();
        await using var connection = new SqliteConnection($"Data Source={_dbPath}");
        await connection.OpenAsync(cancellationToken);

        var photoMap = new Dictionary<string, PhotoItem>(StringComparer.OrdinalIgnoreCase);
        var photoCmd = connection.CreateCommand();
        photoCmd.CommandText = """
            SELECT Id, LibraryFolder, ThumbnailPath, Path, FileName, ImportedAt, CapturedAt, Iso, Aperture, ShutterSpeed, FocalLength, WhiteBalance, CameraMake, CameraModel, LensModel
            FROM Image
            ORDER BY ImportedAt DESC;
            """;

        await using (var reader = await photoCmd.ExecuteReaderAsync(cancellationToken))
        {
            while (await reader.ReadAsync(cancellationToken))
            {
                var id = reader.GetString(0);
                var item = new PhotoItem
                {
                    Id = Guid.TryParse(id, out var parsed) ? parsed : Guid.NewGuid(),
                    LibraryFolder = reader.IsDBNull(1) ? string.Empty : reader.GetString(1),
                    ThumbnailPath = reader.IsDBNull(2) ? string.Empty : reader.GetString(2),
                    Path = reader.GetString(3),
                    FileName = reader.GetString(4),
                    ImportedAt = DateTimeOffset.Parse(reader.GetString(5)),
                    Metadata = new PhotoMetadata
                    {
                        CapturedAt = reader.IsDBNull(6) ? null : DateTimeOffset.Parse(reader.GetString(6)),
                        Iso = reader.IsDBNull(7) ? null : reader.GetInt32(7),
                        Aperture = reader.IsDBNull(8) ? null : reader.GetString(8),
                        ShutterSpeed = reader.IsDBNull(9) ? null : reader.GetString(9),
                        FocalLength = reader.IsDBNull(10) ? null : reader.GetString(10),
                        WhiteBalance = reader.IsDBNull(11) ? null : reader.GetString(11),
                        CameraMake = reader.IsDBNull(12) ? null : reader.GetString(12),
                        CameraModel = reader.IsDBNull(13) ? null : reader.GetString(13),
                        LensModel = reader.IsDBNull(14) ? null : reader.GetString(14),
                    }
                };
                result.Add(item);
                photoMap[item.Id.ToString()] = item;
            }
        }

        var analysisCmd = connection.CreateCommand();
        analysisCmd.CommandText = """
            SELECT ImageId, OverallScore, SharpnessScore, ExposureScore, EyesClosed, IsDuplicate, IsAnalyzed, FaceCount, PersonLabel, StyleLabel, ColorLabel, DominantColorsJson, AutoClass, IsWaste, WasteReason, Rating, IsPicked, RotationQuarterTurns, RawJson
            FROM Analysis;
            """;
        await using (var reader = await analysisCmd.ExecuteReaderAsync(cancellationToken))
        {
            while (await reader.ReadAsync(cancellationToken))
            {
                var id = reader.GetString(0);
                if (!photoMap.TryGetValue(id, out var item))
                {
                    continue;
                }

                item.Analysis = new AnalysisAggregate
                {
                    OverallScore = reader.IsDBNull(1) ? 0 : (float)reader.GetDouble(1),
                    SharpnessScore = reader.IsDBNull(2) ? 0 : (float)reader.GetDouble(2),
                    ExposureScore = reader.IsDBNull(3) ? 0 : (float)reader.GetDouble(3),
                    EyesClosed = !reader.IsDBNull(4) && reader.GetInt32(4) == 1,
                    IsDuplicate = !reader.IsDBNull(5) && reader.GetInt32(5) == 1,
                    IsAnalyzed = !reader.IsDBNull(6) && reader.GetInt32(6) == 1,
                    FaceCount = reader.IsDBNull(7) ? 0 : reader.GetInt32(7),
                    PersonLabel = reader.IsDBNull(8) ? "none" : reader.GetString(8),
                    StyleLabel = reader.IsDBNull(9) ? "unknown" : reader.GetString(9),
                    ColorLabel = reader.IsDBNull(10) ? "unknown" : reader.GetString(10),
                    DominantColors = ParseDominantColors(reader.IsDBNull(11) ? string.Empty : reader.GetString(11)),
                    AutoClass = reader.IsDBNull(12) ? "unknown" : reader.GetString(12),
                    IsWaste = !reader.IsDBNull(13) && reader.GetInt32(13) == 1,
                    WasteReason = reader.IsDBNull(14) ? string.Empty : reader.GetString(14),
                    Rating = reader.IsDBNull(15) ? 0 : reader.GetInt32(15),
                    IsPicked = !reader.IsDBNull(16) && reader.GetInt32(16) == 1,
                    RotationQuarterTurns = reader.IsDBNull(17) ? 0 : NormalizeQuarterTurns(reader.GetInt32(17)),
                    RawJson = reader.IsDBNull(18) ? string.Empty : reader.GetString(18)
                };
            }
        }

        var pluginCmd = connection.CreateCommand();
        pluginCmd.CommandText = """
            SELECT ImageId, PluginName, Score, Payload
            FROM PluginResult;
            """;
        await using var pluginReader = await pluginCmd.ExecuteReaderAsync(cancellationToken);
        while (await pluginReader.ReadAsync(cancellationToken))
        {
            var id = pluginReader.GetString(0);
            if (!photoMap.TryGetValue(id, out var item))
            {
                continue;
            }

            var payload = pluginReader.IsDBNull(3) ? "{}" : pluginReader.GetString(3);
            var parsed = ParsePluginPayload(payload);
            item.Analysis.PluginResults.Add(new PluginResult
            {
                PluginName = pluginReader.GetString(1),
                Score = pluginReader.IsDBNull(2) ? 0 : (float)pluginReader.GetDouble(2),
                Features = parsed.Features,
                Objects = parsed.Objects
            });
        }

        return result;
    }

    private static async Task UpsertPhotoAsync(
        SqliteConnection connection,
        DbTransaction tx,
        PhotoItem photo,
        CancellationToken cancellationToken)
    {
        var cmd = connection.CreateCommand();
        cmd.Transaction = (SqliteTransaction)tx;
        cmd.CommandText = """
            INSERT INTO Image(Id, LibraryFolder, ThumbnailPath, Path, FileName, ImportedAt, CapturedAt, Iso, Aperture, ShutterSpeed, FocalLength, WhiteBalance, CameraMake, CameraModel, LensModel)
            VALUES ($id, $libraryFolder, $thumbnailPath, $path, $fileName, $importedAt, $capturedAt, $iso, $aperture, $shutter, $focal, $whiteBalance, $cameraMake, $camera, $lens)
            ON CONFLICT(Id) DO UPDATE SET
                LibraryFolder = excluded.LibraryFolder,
                ThumbnailPath = excluded.ThumbnailPath,
                Path = excluded.Path,
                FileName = excluded.FileName,
                ImportedAt = excluded.ImportedAt,
                CapturedAt = excluded.CapturedAt,
                Iso = excluded.Iso,
                Aperture = excluded.Aperture,
                ShutterSpeed = excluded.ShutterSpeed,
                FocalLength = excluded.FocalLength,
                WhiteBalance = excluded.WhiteBalance,
                CameraMake = excluded.CameraMake,
                CameraModel = excluded.CameraModel,
                LensModel = excluded.LensModel;
            """;
        cmd.Parameters.AddWithValue("$id", photo.Id.ToString());
        cmd.Parameters.AddWithValue("$libraryFolder", photo.LibraryFolder);
        cmd.Parameters.AddWithValue("$thumbnailPath", (object?)photo.ThumbnailPath ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$path", photo.Path);
        cmd.Parameters.AddWithValue("$fileName", photo.FileName);
        cmd.Parameters.AddWithValue("$importedAt", photo.ImportedAt.ToString("O"));
        cmd.Parameters.AddWithValue("$capturedAt", (object?)photo.Metadata.CapturedAt?.ToString("O") ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$iso", (object?)photo.Metadata.Iso ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$aperture", (object?)photo.Metadata.Aperture ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$shutter", (object?)photo.Metadata.ShutterSpeed ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$focal", (object?)photo.Metadata.FocalLength ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$whiteBalance", (object?)photo.Metadata.WhiteBalance ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$cameraMake", (object?)photo.Metadata.CameraMake ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$camera", (object?)photo.Metadata.CameraModel ?? DBNull.Value);
        cmd.Parameters.AddWithValue("$lens", (object?)photo.Metadata.LensModel ?? DBNull.Value);
        await cmd.ExecuteNonQueryAsync(cancellationToken);
    }

    private static async Task UpsertAnalysisAsync(
        SqliteConnection connection,
        DbTransaction tx,
        PhotoItem photo,
        CancellationToken cancellationToken)
    {
        var cmd = connection.CreateCommand();
        cmd.Transaction = (SqliteTransaction)tx;
        cmd.CommandText = """
            INSERT INTO Analysis(ImageId, OverallScore, SharpnessScore, ExposureScore, EyesClosed, IsDuplicate, IsAnalyzed, FaceCount, PersonLabel, StyleLabel, ColorLabel, DominantColorsJson, AutoClass, IsWaste, WasteReason, Rating, IsPicked, RotationQuarterTurns, RawJson)
            VALUES ($id, $overall, $sharpness, $exposure, $eyesClosed, $duplicate, $isAnalyzed, $faceCount, $personLabel, $styleLabel, $colorLabel, $dominantColorsJson, $autoClass, $isWaste, $wasteReason, $rating, $isPicked, $rotationQuarterTurns, $rawJson)
            ON CONFLICT(ImageId) DO UPDATE SET
                OverallScore = excluded.OverallScore,
                SharpnessScore = excluded.SharpnessScore,
                ExposureScore = excluded.ExposureScore,
                EyesClosed = excluded.EyesClosed,
                IsDuplicate = excluded.IsDuplicate,
                IsAnalyzed = excluded.IsAnalyzed,
                FaceCount = excluded.FaceCount,
                PersonLabel = excluded.PersonLabel,
                StyleLabel = excluded.StyleLabel,
                ColorLabel = excluded.ColorLabel,
                DominantColorsJson = excluded.DominantColorsJson,
                AutoClass = excluded.AutoClass,
                IsWaste = excluded.IsWaste,
                WasteReason = excluded.WasteReason,
                Rating = excluded.Rating,
                IsPicked = excluded.IsPicked,
                RotationQuarterTurns = excluded.RotationQuarterTurns,
                RawJson = excluded.RawJson;
            """;
        cmd.Parameters.AddWithValue("$id", photo.Id.ToString());
        cmd.Parameters.AddWithValue("$overall", photo.Analysis.OverallScore);
        cmd.Parameters.AddWithValue("$sharpness", photo.Analysis.SharpnessScore);
        cmd.Parameters.AddWithValue("$exposure", photo.Analysis.ExposureScore);
        cmd.Parameters.AddWithValue("$eyesClosed", photo.Analysis.EyesClosed ? 1 : 0);
        cmd.Parameters.AddWithValue("$duplicate", photo.Analysis.IsDuplicate ? 1 : 0);
        cmd.Parameters.AddWithValue("$isAnalyzed", photo.Analysis.IsAnalyzed ? 1 : 0);
        cmd.Parameters.AddWithValue("$faceCount", photo.Analysis.FaceCount);
        cmd.Parameters.AddWithValue("$personLabel", photo.Analysis.PersonLabel);
        cmd.Parameters.AddWithValue("$styleLabel", photo.Analysis.StyleLabel);
        cmd.Parameters.AddWithValue("$colorLabel", photo.Analysis.ColorLabel);
        cmd.Parameters.AddWithValue("$dominantColorsJson", SerializeDominantColors(photo.Analysis.DominantColors));
        cmd.Parameters.AddWithValue("$autoClass", photo.Analysis.AutoClass);
        cmd.Parameters.AddWithValue("$isWaste", photo.Analysis.IsWaste ? 1 : 0);
        cmd.Parameters.AddWithValue("$wasteReason", photo.Analysis.WasteReason);
        cmd.Parameters.AddWithValue("$rating", photo.Analysis.Rating);
        cmd.Parameters.AddWithValue("$isPicked", photo.Analysis.IsPicked ? 1 : 0);
        cmd.Parameters.AddWithValue("$rotationQuarterTurns", NormalizeQuarterTurns(photo.Analysis.RotationQuarterTurns));
        cmd.Parameters.AddWithValue("$rawJson", photo.Analysis.RawJson);
        await cmd.ExecuteNonQueryAsync(cancellationToken);

        var deleteCmd = connection.CreateCommand();
        deleteCmd.Transaction = (SqliteTransaction)tx;
        deleteCmd.CommandText = "DELETE FROM PluginResult WHERE ImageId = $id;";
        deleteCmd.Parameters.AddWithValue("$id", photo.Id.ToString());
        await deleteCmd.ExecuteNonQueryAsync(cancellationToken);

        foreach (var plugin in photo.Analysis.PluginResults)
        {
            var pluginCmd = connection.CreateCommand();
            pluginCmd.Transaction = (SqliteTransaction)tx;
            pluginCmd.CommandText = """
                INSERT INTO PluginResult(ImageId, PluginName, Score, Payload)
                VALUES ($id, $name, $score, $payload);
                """;
            var payload = JsonSerializer.Serialize(new
            {
                plugin.Features,
                plugin.Objects
            });
            pluginCmd.Parameters.AddWithValue("$id", photo.Id.ToString());
            pluginCmd.Parameters.AddWithValue("$name", plugin.PluginName);
            pluginCmd.Parameters.AddWithValue("$score", plugin.Score);
            pluginCmd.Parameters.AddWithValue("$payload", payload);
            await pluginCmd.ExecuteNonQueryAsync(cancellationToken);
        }
    }

    private static void EnsureSchema(SqliteConnection connection)
    {
        var cmd = connection.CreateCommand();
        cmd.CommandText = """
            CREATE TABLE IF NOT EXISTS Image (
                Id TEXT PRIMARY KEY,
                LibraryFolder TEXT NULL,
                ThumbnailPath TEXT NULL,
                Path TEXT NOT NULL,
                FileName TEXT NOT NULL,
                ImportedAt TEXT NOT NULL,
                CapturedAt TEXT NULL,
                Iso INTEGER NULL,
                Aperture TEXT NULL,
                ShutterSpeed TEXT NULL,
                FocalLength TEXT NULL,
                WhiteBalance TEXT NULL,
                CameraMake TEXT NULL,
                CameraModel TEXT NULL,
                LensModel TEXT NULL
            );

            CREATE TABLE IF NOT EXISTS Analysis (
                ImageId TEXT PRIMARY KEY,
                OverallScore REAL,
                SharpnessScore REAL,
                ExposureScore REAL,
                EyesClosed INTEGER,
                IsDuplicate INTEGER,
                IsAnalyzed INTEGER,
                FaceCount INTEGER,
                PersonLabel TEXT,
                StyleLabel TEXT,
                ColorLabel TEXT,
                DominantColorsJson TEXT,
                AutoClass TEXT,
                IsWaste INTEGER,
                WasteReason TEXT,
                Rating INTEGER,
                IsPicked INTEGER,
                RotationQuarterTurns INTEGER,
                RawJson TEXT,
                FOREIGN KEY(ImageId) REFERENCES Image(Id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS PluginResult (
                Id INTEGER PRIMARY KEY AUTOINCREMENT,
                ImageId TEXT NOT NULL,
                PluginName TEXT NOT NULL,
                Score REAL,
                Payload TEXT,
                FOREIGN KEY(ImageId) REFERENCES Image(Id) ON DELETE CASCADE
            );
            """;
        cmd.ExecuteNonQuery();
        EnsureColumn(connection, "Image", "CapturedAt", "TEXT NULL");
        EnsureColumn(connection, "Image", "LibraryFolder", "TEXT NULL");
        EnsureColumn(connection, "Image", "ThumbnailPath", "TEXT NULL");
        EnsureColumn(connection, "Image", "Iso", "INTEGER NULL");
        EnsureColumn(connection, "Image", "Aperture", "TEXT NULL");
        EnsureColumn(connection, "Image", "ShutterSpeed", "TEXT NULL");
        EnsureColumn(connection, "Image", "FocalLength", "TEXT NULL");
        EnsureColumn(connection, "Image", "WhiteBalance", "TEXT NULL");
        EnsureColumn(connection, "Image", "CameraMake", "TEXT NULL");
        EnsureColumn(connection, "Image", "CameraModel", "TEXT NULL");
        EnsureColumn(connection, "Image", "LensModel", "TEXT NULL");
        EnsureColumn(connection, "Analysis", "IsAnalyzed", "INTEGER");
        EnsureColumn(connection, "Analysis", "FaceCount", "INTEGER");
        EnsureColumn(connection, "Analysis", "PersonLabel", "TEXT");
        EnsureColumn(connection, "Analysis", "StyleLabel", "TEXT");
        EnsureColumn(connection, "Analysis", "ColorLabel", "TEXT");
        EnsureColumn(connection, "Analysis", "DominantColorsJson", "TEXT");
        EnsureColumn(connection, "Analysis", "AutoClass", "TEXT");
        EnsureColumn(connection, "Analysis", "IsWaste", "INTEGER");
        EnsureColumn(connection, "Analysis", "WasteReason", "TEXT");
        EnsureColumn(connection, "Analysis", "Rating", "INTEGER");
        EnsureColumn(connection, "Analysis", "IsPicked", "INTEGER");
        EnsureColumn(connection, "Analysis", "RotationQuarterTurns", "INTEGER");
    }

    private static void EnsureColumn(SqliteConnection connection, string table, string column, string columnType)
    {
        var cmd = connection.CreateCommand();
        cmd.CommandText = $"PRAGMA table_info({table});";
        using var reader = cmd.ExecuteReader();
        while (reader.Read())
        {
            if (reader.GetString(1).Equals(column, StringComparison.OrdinalIgnoreCase))
            {
                return;
            }
        }

        var alter = connection.CreateCommand();
        alter.CommandText = $"ALTER TABLE {table} ADD COLUMN {column} {columnType};";
        alter.ExecuteNonQuery();
    }

    private static (Dictionary<string, object> Features, List<DetectedObject> Objects) ParsePluginPayload(string payload)
    {
        try
        {
            var root = JsonSerializer.Deserialize<JsonElement>(payload);
            var features = new Dictionary<string, object>();
            if (root.TryGetProperty("features", out var fNode) && fNode.ValueKind == JsonValueKind.Object)
            {
                foreach (var prop in fNode.EnumerateObject())
                {
                    features[prop.Name] = prop.Value.ToString();
                }
            }

            var objects = new List<DetectedObject>();
            if (root.TryGetProperty("objects", out var oNode) && oNode.ValueKind == JsonValueKind.Array)
            {
                foreach (var item in oNode.EnumerateArray())
                {
                    objects.Add(new DetectedObject
                    {
                        Label = item.TryGetProperty("label", out var label) ? label.GetString() ?? string.Empty : string.Empty,
                        Confidence = item.TryGetProperty("confidence", out var conf) ? conf.GetSingle() : 0,
                        X1 = item.TryGetProperty("x1", out var x1) ? x1.GetSingle() : 0,
                        Y1 = item.TryGetProperty("y1", out var y1) ? y1.GetSingle() : 0,
                        X2 = item.TryGetProperty("x2", out var x2) ? x2.GetSingle() : 0,
                        Y2 = item.TryGetProperty("y2", out var y2) ? y2.GetSingle() : 0,
                    });
                }
            }

            return (features, objects);
        }
        catch
        {
            return (new Dictionary<string, object>(), new List<DetectedObject>());
        }
    }

    private static string SerializeDominantColors(IReadOnlyCollection<string> colors)
    {
        return JsonSerializer.Serialize(colors);
    }

    private static List<string> ParseDominantColors(string input)
    {
        if (string.IsNullOrWhiteSpace(input))
        {
            return new List<string>();
        }

        try
        {
            return JsonSerializer.Deserialize<List<string>>(input) ?? new List<string>();
        }
        catch
        {
            return new List<string>();
        }
    }

    private static int NormalizeQuarterTurns(int turns)
    {
        var normalized = turns % 4;
        return normalized < 0 ? normalized + 4 : normalized;
    }
}

