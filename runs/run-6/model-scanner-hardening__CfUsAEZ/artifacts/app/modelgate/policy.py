"""Capabilities from the registry policy, not suspicious words in model data."""

TF_OPS = {}
for kind, names in {
    "code-execution": "PyFunc PyFuncStateless EagerPyFunc",
    "file-access": """ReadFile WriteFile MatchingFiles MatchingFilesDataset ImmutableConst
        WholeFileReader WholeFileReaderV2 TextLineReader TextLineReaderV2
        FixedLengthRecordReader FixedLengthRecordReaderV2 TFRecordReader TFRecordReaderV2
        LMDBReader LMDBDataset ExperimentalLMDBDataset TextLineDataset TFRecordDataset
        FixedLengthRecordDataset DatasetToTFRecord ExperimentalDatasetToTFRecord
        SaveDataset SaveDatasetV2 LoadDataset SnapshotDataset SnapshotDatasetV2
        LoadAndRemapMatrix GenerateVocabRemapping Save SaveSlices Restore RestoreSlice""",
    "network": """DataServiceDataset DataServiceDatasetV2 DataServiceDatasetV3
        DataServiceDatasetV4 RegisterDataset RegisterDatasetV2 DistributedSave""",
}.items():
    TF_OPS.update(dict.fromkeys(names.split(), kind))
