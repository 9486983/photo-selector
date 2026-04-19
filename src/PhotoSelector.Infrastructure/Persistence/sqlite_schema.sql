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
