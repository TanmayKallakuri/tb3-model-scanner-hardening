"""TensorFlow op table (registry policy section 4, closed).

An op listed here has the capability listed; an op not listed has none,
wherever it appears.  Op names are exact and case-sensitive.
"""

CODE = "code-execution"
FILE = "file-access"
NET = "network"

TF_OPS = {}
for _op in ("PyFunc", "PyFuncStateless", "EagerPyFunc"):
    TF_OPS[_op] = CODE
for _op in ("ReadFile", "WriteFile", "MatchingFiles", "MatchingFilesDataset", "ImmutableConst",
            "WholeFileReader", "WholeFileReaderV2", "TextLineReader", "TextLineReaderV2",
            "FixedLengthRecordReader", "FixedLengthRecordReaderV2", "TFRecordReader",
            "TFRecordReaderV2", "LMDBReader", "LMDBDataset", "ExperimentalLMDBDataset",
            "TextLineDataset", "TFRecordDataset", "FixedLengthRecordDataset", "DatasetToTFRecord",
            "ExperimentalDatasetToTFRecord", "SaveDataset", "SaveDatasetV2", "LoadDataset",
            "SnapshotDataset", "SnapshotDatasetV2", "LoadAndRemapMatrix", "GenerateVocabRemapping",
            "Save", "SaveSlices", "Restore", "RestoreSlice"):
    TF_OPS[_op] = FILE
for _op in ("DataServiceDataset", "DataServiceDatasetV2", "DataServiceDatasetV3",
            "DataServiceDatasetV4", "RegisterDataset", "RegisterDatasetV2", "DistributedSave"):
    TF_OPS[_op] = NET
