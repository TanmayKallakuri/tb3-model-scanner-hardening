"""The closed TensorFlow capability table from the registry policy.

Python capabilities are evaluated at call sites by the abstract interpreter;
identifiers and string literals are never findings on their own.
"""
TF_OPS = {}
for _kind, _ops in {
    'code-execution': 'PyFunc PyFuncStateless EagerPyFunc',
    'file-access': '''ReadFile WriteFile MatchingFiles MatchingFilesDataset
        ImmutableConst WholeFileReader WholeFileReaderV2 TextLineReader
        TextLineReaderV2 FixedLengthRecordReader FixedLengthRecordReaderV2
        TFRecordReader TFRecordReaderV2 LMDBReader LMDBDataset
        ExperimentalLMDBDataset TextLineDataset TFRecordDataset
        FixedLengthRecordDataset DatasetToTFRecord ExperimentalDatasetToTFRecord
        SaveDataset SaveDatasetV2 LoadDataset SnapshotDataset SnapshotDatasetV2
        LoadAndRemapMatrix GenerateVocabRemapping Save SaveSlices Restore RestoreSlice''',
    'network': '''DataServiceDataset DataServiceDatasetV2 DataServiceDatasetV3
        DataServiceDatasetV4 RegisterDataset RegisterDatasetV2 DistributedSave''',
}.items():
    TF_OPS.update(dict.fromkeys(_ops.split(), _kind))
