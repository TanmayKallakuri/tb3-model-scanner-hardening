"""Format-level policy tables (see /app/docs/threat-model.md sections 4 and 5)."""

# TensorFlow: the op table is closed.  An op listed here has the capability
# listed; any other op (SaveV2, RestoreV2, MergeV2Checkpoints, ...) has none.
TF_OPS = {}
for _op in ("PyFunc", "PyFuncStateless", "EagerPyFunc"):
    TF_OPS[_op] = "code-execution"
for _op in ("ReadFile", "WriteFile", "MatchingFiles", "MatchingFilesDataset", "ImmutableConst", "WholeFileReader",
            "WholeFileReaderV2", "TextLineReader", "TextLineReaderV2", "FixedLengthRecordReader",
            "FixedLengthRecordReaderV2", "TFRecordReader", "TFRecordReaderV2", "LMDBReader", "LMDBDataset",
            "ExperimentalLMDBDataset", "TextLineDataset", "TFRecordDataset", "FixedLengthRecordDataset",
            "DatasetToTFRecord", "ExperimentalDatasetToTFRecord", "SaveDataset", "SaveDatasetV2", "LoadDataset",
            "SnapshotDataset", "SnapshotDatasetV2", "LoadAndRemapMatrix", "GenerateVocabRemapping", "Save",
            "SaveSlices", "Restore", "RestoreSlice"):
    TF_OPS[_op] = "file-access"
for _op in ("DataServiceDataset", "DataServiceDatasetV2", "DataServiceDatasetV3", "DataServiceDatasetV4",
            "RegisterDataset", "RegisterDatasetV2", "DistributedSave"):
    TF_OPS[_op] = "network"

# Keras: config keys whose value Keras deserializes as a function/object, so a
# "__lambda__" dict there is materialised from bytecode.
KERAS_LAMBDA_KEYS = ("function", "output_shape", "mask")
KERAS_GENERIC_KEYS = ("activation", "recurrent_activation")
KERAS_GENERIC_SUFFIXES = ("_activation", "_initializer", "_regularizer", "_constraint")
