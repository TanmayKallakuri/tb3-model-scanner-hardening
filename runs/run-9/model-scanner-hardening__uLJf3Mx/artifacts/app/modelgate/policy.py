"""The closed graph-op policy. Pickle capabilities are evaluated at call sites."""

TF_OPS = {}
for _kind, _names in {
    "code-execution": "PyFunc PyFuncStateless EagerPyFunc",
    "file-access": """ReadFile WriteFile MatchingFiles MatchingFilesDataset
        ImmutableConst WholeFileReader WholeFileReaderV2 TextLineReader
        TextLineReaderV2 FixedLengthRecordReader FixedLengthRecordReaderV2
        TFRecordReader TFRecordReaderV2 LMDBReader LMDBDataset
        ExperimentalLMDBDataset TextLineDataset TFRecordDataset
        FixedLengthRecordDataset DatasetToTFRecord ExperimentalDatasetToTFRecord
        SaveDataset SaveDatasetV2 LoadDataset SnapshotDataset SnapshotDatasetV2
        LoadAndRemapMatrix GenerateVocabRemapping Save SaveSlices Restore RestoreSlice""",
    "network": """DataServiceDataset DataServiceDatasetV2 DataServiceDatasetV3
        DataServiceDatasetV4 RegisterDataset RegisterDatasetV2 DistributedSave""",
}.items():
    TF_OPS.update(dict.fromkeys(_names.split(), _kind))
