"""The closed TensorFlow capability table from the registry policy.

Pickle capabilities are interpreted at call sites in pickle_scan, rather than
attached to strings or to the mere presence of a global.
"""
TF_OPS = {name: 'code-execution' for name in (
    'PyFunc', 'PyFuncStateless', 'EagerPyFunc',
)}
TF_OPS.update({name: 'file-access' for name in '''
ReadFile WriteFile MatchingFiles MatchingFilesDataset ImmutableConst
WholeFileReader WholeFileReaderV2 TextLineReader TextLineReaderV2
FixedLengthRecordReader FixedLengthRecordReaderV2 TFRecordReader TFRecordReaderV2
LMDBReader LMDBDataset ExperimentalLMDBDataset TextLineDataset TFRecordDataset
FixedLengthRecordDataset DatasetToTFRecord ExperimentalDatasetToTFRecord
SaveDataset SaveDatasetV2 LoadDataset SnapshotDataset SnapshotDatasetV2
LoadAndRemapMatrix GenerateVocabRemapping Save SaveSlices Restore RestoreSlice
'''.split()})
TF_OPS.update({name: 'network' for name in '''
DataServiceDataset DataServiceDatasetV2 DataServiceDatasetV3 DataServiceDatasetV4
RegisterDataset RegisterDatasetV2 DistributedSave
'''.split()})
