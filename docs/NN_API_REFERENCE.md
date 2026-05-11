# Neural Network Backend API Reference

Version: 1.0  
Date: 2026-05-08

This reference documents the current NN-related Python APIs in ai-resources.

Conventions:

- Type hints mirror current signatures where available
- Examples are minimal and local-first
- Related methods link by class/module name only
- If a method is requested in design docs but missing in code, it is marked clearly

---

## 1. Module: nn_tokenizer.py

### 1.1 Class: PasswordTokenizer

Purpose:

- Convert raw password strings into fixed-length character embeddings

#### Constructor

Signature:

```python
PasswordTokenizer(max_length: int = 32, embedding_dim: int = 8)
```

Parameters:

- max_length (int): maximum characters retained per password
- embedding_dim (int): embedding vector size per character token

Returns:

- PasswordTokenizer instance

Example:

```python
from nn_tokenizer import PasswordTokenizer

tok = PasswordTokenizer(max_length=32, embedding_dim=8)
```

Related:

- PasswordTokenizer.encode
- PasswordTokenizer.batch_encode
- PasswordTokenizer.num_embedding_params

#### Method: encode

Signature:

```python
encode(password: str) -> torch.Tensor
```

Parameters:

- password (str): input password

Return value:

- torch.Tensor with shape (1, max_length, embedding_dim)

Behavior notes:

- Truncates passwords longer than max_length
- Pads shorter passwords with token value 0
- Uses ASCII ord(c) clamped to 255

Example:

```python
t = tok.encode("P@ssw0rd!")
print(t.shape)
```

Related:

- batch_encode
- to

#### Method: batch_encode

Signature:

```python
batch_encode(passwords: List[str]) -> torch.Tensor
```

Parameters:

- passwords (List[str]): batch of strings

Return value:

- torch.Tensor with shape (batch_size, max_length, embedding_dim)
- Returns empty tensor shape (0, max_length, embedding_dim) for empty input list

Example:

```python
batch = tok.batch_encode(["abc", "letmein", "hunter2"])
print(batch.shape)
```

Related:

- encode

#### Property: num_embedding_params

Signature:

```python
num_embedding_params: int
```

Return value:

- 256 * embedding_dim

Example:

```python
print(tok.num_embedding_params)
```

#### Method: to

Signature:

```python
to(device) -> PasswordTokenizer
```

Parameters:

- device: torch device object or compatible target

Return value:

- self (PasswordTokenizer)

Use case:

- Move embedding layer to CPU/GPU with trainer model

---

## 2. Module: nn_models.py

### 2.1 Class: PasswordCNN

Purpose:

- Compact multi-branch CNN binary classifier for password risk

#### Constructor

Signature:

```python
PasswordCNN(embedding_dim: int = 8, hidden_dim: int = 64, dropout: float = 0.3)
```

Parameters:

- embedding_dim (int): tokenizer embedding feature size
- hidden_dim (int): dense hidden layer width
- dropout (float): dropout probability in dense block

Return value:

- PasswordCNN (torch.nn.Module)

Architecture summary:

- Conv1d branches with kernels 2, 3, 4
- 3 filters per branch
- AdaptiveMaxPool1d(8)
- Dense stack: 72 -> hidden_dim -> 1

Example:

```python
from nn_models import PasswordCNN

model = PasswordCNN(embedding_dim=8, hidden_dim=64, dropout=0.2)
```

Related:

- PasswordCNN.forward
- PasswordCNN.num_params

#### Method: forward

Signature:

```python
forward(x: torch.Tensor) -> torch.Tensor
```

Parameters:

- x (torch.Tensor): shape (B, max_length, embedding_dim)

Return value:

- logits tensor shape (B, 1)

Behavior notes:

- Returns logits, not probabilities
- Use torch.sigmoid(logits) for probability
- Intended loss: BCEWithLogitsLoss

Example:

```python
import torch

x = torch.randn(4, 32, 8)
logits = model(x)
probs = torch.sigmoid(logits)
```

Related:

- nn_trainer.PasswordNNTrainer._train_epoch

#### Property: num_params

Signature:

```python
num_params: int
```

Return value:

- Total trainable + non-trainable parameter count from model.parameters()

Example:

```python
print(model.num_params)
```

### 2.2 Class: PasswordCNNConfigurable

Purpose:

- Variant with configurable dense-stack depth and widths

#### Constructor

Signature:

```python
PasswordCNNConfigurable(
    embedding_dim: int = 8,
    hidden_dims: Optional[List[int]] = None,
    dropout: float = 0.3,
)
```

Parameters:

- embedding_dim (int): input embedding dimension
- hidden_dims (Optional[List[int]]): dense layer widths, defaults to [64, 32]
- dropout (float): dropout probability applied after each hidden dense layer

Return value:

- PasswordCNNConfigurable

Example:

```python
from nn_models import PasswordCNNConfigurable

model = PasswordCNNConfigurable(hidden_dims=[128, 64, 32], dropout=0.2)
```

#### Method: forward

Signature:

```python
forward(x: torch.Tensor) -> torch.Tensor
```

Parameters:

- x (torch.Tensor): shape (B, max_length, embedding_dim)

Return value:

- logits tensor shape (B, 1)

Example:

```python
x = torch.randn(8, 32, 8)
logits = model(x)
```

#### Property: num_params

Signature:

```python
num_params: int
```

Return value:

- Parameter count across full module

Related:

- PasswordCNN.num_params

---

## 3. Module: nn_trainer.py

### 3.1 Class: PasswordNNTrainer

Purpose:

- End-to-end training orchestration with tokenizer, model, registry, telemetry, and control signals

#### Constructor

Signature:

```python
PasswordNNTrainer(model_dir: str = "models/nn", device: Optional[torch.device] = None)
```

Parameters:

- model_dir (str): artifact root folder for NNModelRegistry
- device (Optional[torch.device]): explicit device or auto-detected via GPUManager

Return value:

- PasswordNNTrainer instance

Side effects:

- Ensures model_dir exists
- Initializes tokenizer and moves it to device
- Initializes registry

Example:

```python
from nn_trainer import PasswordNNTrainer

trainer = PasswordNNTrainer(model_dir="models/nn")
```

Related:

- GPUManager.detect_device
- NNModelRegistry

#### Method: train

Signature:

```python
train(
    X: pd.Series,
    y: pd.Series,
    epochs: int = 20,
    batch_size: Optional[int] = None,
    learning_rate: float = 0.001,
    val_split: float = 0.2,
    control_signal: Optional[ControlSignal] = None,
    telemetry_emitter: Optional[TelemetryEmitter] = None,
) -> Dict[str, Any]
```

Parameters:

- X (pd.Series): password strings
- y (pd.Series): binary labels (0/1)
- epochs (int): max epoch count
- batch_size (Optional[int]): explicit batch size; auto-selected if None
- learning_rate (float): Adam optimizer lr
- val_split (float): validation fraction
- control_signal (Optional[ControlSignal]): pause/stop/resume state source
- telemetry_emitter (Optional[TelemetryEmitter]): event sink for progress

Return value:

- Dict[str, Any] with keys:
  - model: trained torch model
  - history: dict of epoch, train_loss, val_loss, val_acc arrays
  - best_epoch: int
  - best_metrics: dict with val_loss and val_acc
  - checkpoint_path: str (best model path)
  - training_time_sec: float
  - device: str ("cuda" or "cpu")

Telemetry emitted by current implementation:

- nn.training.started
- nn.epoch.started
- nn.epoch.completed
- nn.training.resumed
- nn.training.stopped
- nn.training.completed

Example:

```python
from shared_lib.control_signal import ControlSignal
from shared_lib.telemetry_emitter import TelemetryEmitter

control = ControlSignal()
emitter = TelemetryEmitter()

result = trainer.train(
    X=df["password"],
    y=df["target"],
    epochs=20,
    batch_size=32,
    learning_rate=1e-3,
    control_signal=control,
    telemetry_emitter=emitter,
)
```

Related:

- _train_epoch
- _validate_epoch
- _train_val_split
- NNModelRegistry.save_checkpoint
- NNModelRegistry.save_best_model

#### Method: load_checkpoint

Status:

- Not present in current PasswordNNTrainer implementation

Requested in design docs:

- load_checkpoint(checkpoint_path)

Current alternative:

- Use NNModelRegistry.load_checkpoint(run_id, epoch)

Compatibility guidance:

- If you need trainer-level checkpoint loading, implement wrapper logic externally by loading state through registry and applying to model/optimizer manually.

#### Method: _train_epoch

Signature:

```python
_train_epoch(model, train_loader, optimizer, loss_fn) -> float
```

Parameters:

- model: torch module
- train_loader: torch DataLoader
- optimizer: torch optimizer
- loss_fn: loss callable

Return value:

- average training loss for epoch (float)

#### Method: _validate_epoch

Signature:

```python
_validate_epoch(model, val_loader, loss_fn) -> Tuple[float, float]
```

Parameters:

- model: torch module in eval mode
- val_loader: DataLoader
- loss_fn: loss callable

Return value:

- (avg_val_loss, accuracy)

Behavior notes:

- Applies sigmoid and threshold 0.5 for binary accuracy

#### Method: _train_val_split

Signature:

```python
_train_val_split(X, y, val_split)
```

Parameters:

- X: sequence-like pandas object
- y: sequence-like pandas object
- val_split: float fraction for validation

Return value:

- (X_train, X_val, y_train, y_val)

---

### 3.2 Class: GPUManager

Purpose:

- Device and runtime utility methods for trainer defaults

#### Method: detect_device

Signature:

```python
detect_device() -> torch.device
```

Return value:

- torch.device("cuda") if available, else torch.device("cpu")

Example:

```python
from nn_trainer import GPUManager

device = GPUManager.detect_device()
```

#### Method: get_batch_size

Signature:

```python
get_batch_size(device: torch.device, fallback: int = 32) -> int
```

Parameters:

- device (torch.device): selected device
- fallback (int): CPU batch size default

Return value:

- 128 for CUDA
- fallback for CPU

Example:

```python
bs = GPUManager.get_batch_size(device)
```

#### Method: get_mixed_precision_context

Signature:

```python
get_mixed_precision_context(device: torch.device)
```

Parameters:

- device (torch.device)

Return value:

- torch.cuda.amp.autocast() context for CUDA
- contextlib.nullcontext() for CPU

Example:

```python
ctx = GPUManager.get_mixed_precision_context(device)
with ctx:
    logits = model(batch_x)
```

---

## 4. Module: nn_registry.py

### 4.1 Class: NNModelRegistry

Purpose:

- Manage NN checkpoints, final model, metadata, and metrics history

#### Constructor

Signature:

```python
NNModelRegistry(registry_dir: str = "models/nn")
```

Parameters:

- registry_dir (str): root NN artifact directory

Behavior:

- Creates directories:
  - checkpoints
  - final
  - history

#### Method: save_checkpoint

Signature:

```python
save_checkpoint(
    run_id: str,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    metrics: Dict[str, Any],
) -> str
```

Parameters:

- run_id (str): run identifier
- epoch (int): epoch index
- model: model instance
- optimizer: optimizer instance
- metrics (dict): epoch metrics

Return value:

- checkpoint file path (str)

Behavior notes:

- Atomic save via temporary file then rename

Example:

```python
path = registry.save_checkpoint(
    run_id="abc123",
    epoch=3,
    model=model,
    optimizer=optimizer,
    metrics={"train_loss": 0.21, "val_loss": 0.22, "val_acc": 0.89},
)
```

Related:

- load_checkpoint

#### Method: load_checkpoint

Signature:

```python
load_checkpoint(run_id: str, epoch: int) -> Dict[str, Any]
```

Parameters:

- run_id (str)
- epoch (int)

Return value:

- Dict with keys: epoch, model_state, optimizer_state, metrics

Raises:

- FileNotFoundError if checkpoint missing

#### Method: save_best_model

Signature:

```python
save_best_model(
    run_id: str,
    model: torch.nn.Module,
    architecture_config: Dict[str, Any],
    metrics: Dict[str, Any],
    training_config: Dict[str, Any],
) -> str
```

Parameters:

- run_id (str)
- model (torch.nn.Module)
- architecture_config (dict)
- metrics (dict)
- training_config (dict)

Return value:

- best model path (str)

Artifacts produced:

- final/{run_id}/best_model.pt
- final/{run_id}/metadata.json

#### Method: load_best_model

Signature:

```python
load_best_model(run_id: str) -> Tuple[torch.nn.Module, Dict[str, Any]]
```

Parameters:

- run_id (str)

Return value:

- tuple(model, metadata)

Behavior notes:

- Reconstructs PasswordCNN from metadata architecture values

Raises:

- FileNotFoundError if artifacts missing

#### Method: save_history

Signature:

```python
save_history(run_id: str, history: Dict[str, Any]) -> str
```

Parameters:

- run_id (str)
- history (dict): training history arrays

Return value:

- CSV path (str)

#### Method: load_history

Signature:

```python
load_history(run_id: str) -> pd.DataFrame
```

Parameters:

- run_id (str)

Return value:

- pandas DataFrame from history CSV

Raises:

- FileNotFoundError if history missing

---

## 5. Module: ensemble.py

### 5.1 Class: EnsemblePredictor

Purpose:

- Combine Phase 1, Phase 2, and NN predictions via soft voting

#### Constructor

Signature:

```python
EnsemblePredictor()
```

Initial state:

- phase1_model = None
- phase2_model = None
- nn_model = None
- tokenizer = None
- enabled_models flags default to False

#### Method: load_phase1_model

Signature:

```python
load_phase1_model(checkpoint_path: str)
```

Parameters:

- checkpoint_path (str): joblib model path

Return value:

- None

Side effects:

- Loads model
- Sets enabled_models["phase1"] = True

#### Method: load_phase2_model

Signature:

```python
load_phase2_model(checkpoint_path: str)
```

Parameters:

- checkpoint_path (str): joblib model path

Return value:

- None

Side effects:

- Loads model
- Sets enabled_models["phase2"] = True

#### Method: load_nn_model

Signature:

```python
load_nn_model(checkpoint_path: str)
```

Parameters:

- checkpoint_path (str): torch state_dict checkpoint path

Return value:

- None

Behavior notes:

- Instantiates PasswordCNN
- Loads state dict on CPU map_location
- Sets model eval mode
- Creates PasswordTokenizer
- Sets enabled_models["nn"] = True

#### Method: predict_proba

Signature:

```python
predict_proba(passwords: pd.Series, ensemble_method: str = "soft_vote") -> np.ndarray
```

Parameters:

- passwords (pd.Series): input passwords
- ensemble_method (str): currently soft_vote behavior

Return value:

- np.ndarray shape (n_samples, 2): [prob_not_hacked, prob_hacked]

Behavior notes:

- Uses equal weights per active model
- Normalizes weights to 1
- Skips model errors silently
- Raises ValueError if no model contributes

Example:

```python
import pandas as pd

pw = pd.Series(["admin", "hunter2", "S3cure!"])
proba = ensemble.predict_proba(pw)
```

Related:

- predict
- ensemble_config

#### Method: predict

Signature:

```python
predict(
    passwords: pd.Series,
    ensemble_method: str = "soft_vote",
    threshold: float = 0.5,
) -> np.ndarray
```

Parameters:

- passwords (pd.Series)
- ensemble_method (str)
- threshold (float): hacked cutoff

Return value:

- np.ndarray int labels (0/1)

Behavior:

- Computes predict_proba and thresholds column index 1

#### Method: get_active_model_count

Signature:

```python
get_active_model_count() -> int
```

Return value:

- number of enabled models

#### Property: ensemble_config

Signature:

```python
ensemble_config -> dict
```

Return value:

- dict with keys:
  - phase1
  - phase2
  - nn
  - method
  - active_count

---

## 6. Module: model_registry.py

### 6.1 Class: UnifiedModelRegistry

Purpose:

- Discover and query artifacts across phase1, phase2, nn model folders

#### Constructor

Signature:

```python
UnifiedModelRegistry(base_dir: str = "models")
```

Parameters:

- base_dir (str): base model root directory

Return value:

- UnifiedModelRegistry

#### Method: scan_models

Signature:

```python
scan_models() -> Dict[str, List[Dict]]
```

Return value:

- dict keyed by phase with list of discovered best model descriptors:
  - run_id
  - path

Discovery rules:

- phase1 final/**/best_model.joblib
- phase2 final/**/best_model.joblib
- nn final/**/best_model.pt

#### Method: get_latest_models

Signature:

```python
get_latest_models() -> Dict[str, Dict]
```

Return value:

- one latest model entry per phase when available

Selection rule:

- sort by run_id descending, take first

#### Method: get_model_metadata

Signature:

```python
get_model_metadata(phase: str, run_id: str) -> dict
```

Parameters:

- phase (str): phase1, phase2, or nn
- run_id (str): run identifier

Return value:

- metadata dict parsed from metadata.json
- empty dict if metadata missing

---

## 7. Module: model_comparison.py

### 7.1 Class: ModelComparison

Purpose:

- Build table and visual comparison across latest models from each phase

#### Constructor

Signature:

```python
ModelComparison(registry: UnifiedModelRegistry)
```

Parameters:

- registry (UnifiedModelRegistry): model discovery and metadata provider

State:

- self.models initialized from registry.get_latest_models()
- self.comparison_df initialized to None

#### Method: build_comparison_table

Signature:

```python
build_comparison_table() -> pd.DataFrame
```

Return value:

- DataFrame containing rows per phase with columns:
  - Phase
  - Run ID
  - Path
  - Accuracy
  - Precision
  - Recall
  - F1
  - Params
  - Training Time

Data source:

- metadata from registry.get_model_metadata for each latest model

Example:

```python
registry = UnifiedModelRegistry("models")
comp = ModelComparison(registry)
df = comp.build_comparison_table()
print(df)
```

Related:

- get_best_model_by_metric
- plot_comparison

#### Method: get_best_model_by_metric

Signature:

```python
get_best_model_by_metric(metric: str = "accuracy") -> Optional[str]
```

Parameters:

- metric (str): metric name expected to map to capitalized DataFrame column

Return value:

- phase string from Phase column (for max metric)
- None if table empty or column unavailable

Notes:

- metric string is transformed with capitalize(), so use names like accuracy, precision, recall, f1

#### Method: plot_comparison

Signature:

```python
plot_comparison()
```

Return value:

- plotly.graph_objects.Figure

Behavior:

- Creates grouped bar chart for Accuracy, Precision, Recall, F1

---

## 8. Practical Cross-Module Example

```python
import pandas as pd
from nn_trainer import PasswordNNTrainer
from model_registry import UnifiedModelRegistry
from model_comparison import ModelComparison
from ensemble import EnsemblePredictor

# 1) Train NN
trainer = PasswordNNTrainer(model_dir="models/nn")
df = pd.read_csv("ai-resources/data/combined_data.csv")
result = trainer.train(df["password"], df["target"], epochs=10)

# 2) Compare latest models
registry = UnifiedModelRegistry(base_dir="models")
comparison = ModelComparison(registry)
comparison_df = comparison.build_comparison_table()
print(comparison_df)

# 3) Optional ensemble prediction
ensemble = EnsemblePredictor()
ensemble.load_nn_model(result["checkpoint_path"])
proba = ensemble.predict_proba(pd.Series(["admin123", "P@ssw0rd!"]))
print(proba)
```

---

## 9. API Differences vs Planned Interface

Planned and requested entries not fully available as concrete methods today:

- PasswordNNTrainer.load_checkpoint(checkpoint_path): not implemented

Planned telemetry event in design docs with partial/no runtime emission in current trainer:

- nn.training.paused event name exists in architecture/telemetry docs, but current PasswordNNTrainer emits resume/stop/start/epoch/completed events directly.

Recommendation:

- Use this document as source of truth for current runtime API.
- Treat missing planned methods/events as roadmap items, not guaranteed interfaces.
