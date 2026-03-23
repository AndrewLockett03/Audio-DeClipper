import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import optuna
from optuna.trial import TrialState
from pathlib import Path
import os
from dataset import DeclipDataset
from preprocess import chunk_dataset

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR   = "/home/andrew/Documents/data/reduced-train"
CHUNK_DIR  = "/home/andrew/Documents/data/reduced-chunks"
STUDY_NAME = "declip_cnn_study"
STORAGE    = f"sqlite:///{STUDY_NAME}.db"   # persists results across runs

# ── Fixed settings ─────────────────────────────────────────────────────────────
CHUNK_SIZE      = 16384
EPOCHS_PER_TRIAL = 3          # keep short so Optuna can explore more trials
N_TRIALS        = 30          # total Optuna trials to run
VAL_SPLIT       = 0.1         # fraction of chunks used for validation
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Dynamic model that Optuna can configure ────────────────────────────────────
class DeclipCNN(nn.Module):
    """
    Configurable 1-D CNN whose width, depth, and kernel size are
    chosen by the Optuna trial object.
    """
    def __init__(self, n_filters: int, kernel_size: int, n_conv_layers: int):
        super().__init__()

        enc_layers = []
        in_ch = 1
        for i in range(n_conv_layers):
            out_ch = n_filters * (2 ** i)          # double channels each layer
            enc_layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size,
                          padding=kernel_size // 2),
                nn.ReLU(),
            ]
            in_ch = out_ch
        enc_layers.append(nn.MaxPool1d(2))
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers = []
        for i in range(n_conv_layers - 1, -1, -1):
            out_ch = n_filters * (2 ** i)
            dec_layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size,
                          padding=kernel_size // 2),
                nn.ReLU(),
            ]
            in_ch = out_ch
        dec_layers.append(
            nn.Conv1d(in_ch, 1, kernel_size=kernel_size,
                      padding=kernel_size // 2)
        )
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = F.interpolate(x, scale_factor=2, mode="linear", align_corners=False)
        x = self.decoder(x)
        return x


# ── One Optuna trial ───────────────────────────────────────────────────────────
def objective(trial: optuna.Trial) -> float:
    """
    Returns the average validation loss for this trial's hyperparameters.
    Lower is better.
    """
    try:
        # ── Suggest hyperparameters ────────────────────────────────────────────────
        lr          = trial.suggest_float("lr",          1e-4, 1e-2, log=True)
        batch_size  = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        n_filters   = trial.suggest_categorical("n_filters",  [32, 64, 128])
        kernel_size = trial.suggest_categorical("kernel_size", [7, 15, 31])
        n_conv_layers = trial.suggest_int("n_conv_layers", 1, 3)
        weight_decay  = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

        # ── Data ───────────────────────────────────────────────────────────────────
        full_dataset = DeclipDataset(CHUNK_DIR)
        val_size  = max(1, int(len(full_dataset) * VAL_SPLIT))
        train_size = len(full_dataset) - val_size
        train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  shuffle=True,  num_workers=2, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                                  shuffle=False, num_workers=2, pin_memory=True)

        # ── Model ──────────────────────────────────────────────────────────────────
        model = DeclipCNN(
            n_filters=n_filters,
            kernel_size=kernel_size,
            n_conv_layers=n_conv_layers,
        ).to(DEVICE)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        # ── Training loop with pruning ─────────────────────────────────────────────
        for epoch in range(EPOCHS_PER_TRIAL):
            model.train()
            for wave_clipped, wave_original in train_loader:
                wave_clipped  = wave_clipped.to(DEVICE)
                wave_original = wave_original.to(DEVICE)

                optimizer.zero_grad()
                output = model(wave_clipped)
                loss   = criterion(output, wave_original)
                loss.backward()
                optimizer.step()

                del wave_clipped, wave_original, output, loss

            # Validation
            model.eval()
            val_loss = 0.0
            n_batches = 0
            with torch.no_grad():
                for wave_clipped, wave_original in val_loader:
                    wave_clipped  = wave_clipped.to(DEVICE)
                    wave_original = wave_original.to(DEVICE)
                    output = model(wave_clipped)
                    val_loss += criterion(output, wave_original).item()
                    n_batches += 1
                    del wave_clipped, wave_original, output

            avg_val_loss = val_loss / max(n_batches, 1)
            print(f"  Trial {trial.number} | Epoch {epoch+1}/{EPOCHS_PER_TRIAL} "
                  f"| val_loss={avg_val_loss:.6f}")

            # Report to Optuna and prune unpromising trials early
            trial.report(avg_val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return avg_val_loss
    except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise optuna.exceptions.TrialPruned()  # marks trial as pruned, not failed


# ── Re-train best model for full EPOCHS ───────────────────────────────────────
def retrain_best(best_params: dict, full_epochs: int = 8) -> None:
    print("\n=== Retraining best model for full epochs ===")
    print("Best params:", best_params)

    dataset    = DeclipDataset(CHUNK_DIR)
    dataloader = DataLoader(dataset,
                            batch_size=best_params["batch_size"],
                            shuffle=True, num_workers=2, pin_memory=True)

    model = DeclipCNN(
        n_filters=best_params["n_filters"],
        kernel_size=best_params["kernel_size"],
        n_conv_layers=best_params["n_conv_layers"],
    ).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=best_params["lr"],
                           weight_decay=best_params["weight_decay"])

    os.makedirs("best_model_checkpoints", exist_ok=True)

    for epoch in range(full_epochs):
        print(f"BEGINNING EPOCH {epoch + 1}")
        epoch_loss = 0.0
        for wave_clipped, wave_original in dataloader:
            wave_clipped  = wave_clipped.to(DEVICE)
            wave_original = wave_original.to(DEVICE)

            optimizer.zero_grad()
            output = model(wave_clipped)
            loss   = criterion(output, wave_original)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            del wave_clipped, wave_original, output, loss

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{full_epochs}], Loss: {avg_loss:.6f}")

        ckpt = f"best_model_checkpoints/BEST_EPOCH_{epoch+1}.pth"
        torch.save(model.state_dict(), ckpt)

        with open("best_model_run_results.txt", "a") as f:
            f.write(f"Epoch {epoch+1}, Loss: {avg_loss}\n")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # Chunk dataset if needed
    if not Path(CHUNK_DIR).exists():
        print("CHUNKING DATASET")
        chunk_dataset(DATA_DIR, CHUNK_DIR, CHUNK_SIZE)

    # Create (or load existing) Optuna study.
    # TPESampler is Optuna's default Bayesian-style optimizer.
    # MedianPruner stops unpromising trials early.
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5,
                                           n_warmup_steps=1),
        load_if_exists=True,       # resume if the .db already exists
    )

    study.optimize(objective, n_trials=N_TRIALS, timeout=None,
                   gc_after_trial=True)

    # ── Summary ────────────────────────────────────────────────────────────────
    pruned    = len(study.get_trials(deepcopy=False, states=[TrialState.PRUNED]))
    complete  = len(study.get_trials(deepcopy=False, states=[TrialState.COMPLETE]))

    print(f"\nStudy complete. Finished: {complete} | Pruned: {pruned}")
    print("Best trial:")
    t = study.best_trial
    print(f"  Value (val_loss): {t.value:.6f}")
    for k, v in t.params.items():
        print(f"  {k}: {v}")

    # ── Optionally visualize with optuna-dashboard or plotly ──────────────────
    optuna.visualization.plot_optimization_history(study).show()
    optuna.visualization.plot_param_importances(study).show()

    # ── Re-train with best hyperparameters ────────────────────────────────────
    retrain_best(study.best_trial.params, full_epochs=8)
