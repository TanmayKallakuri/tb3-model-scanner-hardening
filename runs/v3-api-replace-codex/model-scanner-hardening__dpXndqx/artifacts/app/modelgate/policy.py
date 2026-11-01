"""The closed TensorFlow table in the registry policy.

Pickle capabilities are classified at *calls*, by the abstract interpreter in
pickle_scan.py / symbolic.py. Merely mentioning a callable is harmless.
"""
TF_OPS = dict.fromkeys('PyFunc PyFuncStateless EagerPyFunc'.split(), 'code-execution')
TF_OPS.update(dict.fromkeys('''
ReadFile WriteFile MatchingFiles MatchingFilesDataset ImmutableConst
WholeFileReader WholeFileReaderV2 TextLineReader TextLineReaderV2
FixedLengthRecordReader FixedLengthRecordReaderV2 TFRecordReader TFRecordReaderV2
LMDBReader LMDBDataset ExperimentalLMDBDataset TextLineDataset TFRecordDataset
FixedLengthRecordDataset DatasetToTFRecord ExperimentalDatasetToTFRecord
SaveDataset SaveDatasetV2 LoadDataset SnapshotDataset SnapshotDatasetV2
LoadAndRemapMatrix GenerateVocabRemapping Save SaveSlices Restore RestoreSlice
'''.split(), 'file-access'))
TF_OPS.update(dict.fromkeys('''
DataServiceDataset DataServiceDatasetV2 DataServiceDatasetV3 DataServiceDatasetV4
RegisterDataset RegisterDatasetV2 DistributedSave
'''.split(), 'network'))
