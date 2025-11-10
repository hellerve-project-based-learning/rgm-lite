import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.transform import resize

# ------------------------------
# tiny utilities
# ------------------------------

def downsample(img, out_h=16, out_w=16):
    """Anti-aliased 28→16 resize. img: 2D float [0,1]."""
    return resize(img, (out_h, out_w), order=1, anti_aliasing=True,
                  preserve_range=True, mode='reflect').astype(np.float32)

def extract_patches(img, patch=4):
    H, W = img.shape; ph = pw = patch
    rows, cols = H // ph, W // pw
    # simple, explicit tiling (easy to read & stitch back)
    patches = []
    for i in range(rows):
        for j in range(cols):
            block = img[i*ph:(i+1)*ph, j*pw:(j+1)*pw]
            patches.append(block.reshape(-1))
    return np.stack(patches, axis=0)  # [rows*cols, ph*pw]

def kmeans(X, K, iters=25, seed=0):
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), K, replace=False)].copy()
    for _ in range(iters):
        z = ((X[:, None, :] - C[None, :, :])**2).sum(2).argmin(1)
        for k in range(K):
            m = (z == k)
            if m.any(): C[k] = X[m].mean(0)
    return C

def standardize(p):
    mu = p.mean(1, keepdims=True); sd = p.std(1, keepdims=True) + 1e-6
    return (p - mu) / sd, mu.squeeze(1), sd.squeeze(1)

# ------------------------------
# model: RGM‑lite (2 levels)
# ------------------------------

class RGMImageClassifier:
    def __init__(self, num_classes=10, K=256, alpha=1.0, beta=1.0, patch=4, bg_threshold=0.10):
        self.C, self.K, self.alpha, self.beta = num_classes, K, float(alpha), float(beta)
        self.patch, self.bg_threshold = patch, float(bg_threshold)
        self.pi  = np.full(self.C, 1.0/self.C)
        self.phi = np.full((self.C, self.K), 1.0/self.K)
        self.codebook = None

    # --- learn a codebook on foreground patches (stroke-centric) ---
    def fit_codebook(self, images_ds, seed=42):
        fg = []
        for x in images_ds:
            p = extract_patches(x, patch=self.patch)
            m = p.mean(1)
            p = p[m > self.bg_threshold]               # keep bright-ish tiles
            if len(p): fg.append(standardize(p)[0])    # standardize before K‑means
        X = np.vstack(fg)
        self.codebook = kmeans(X, self.K, iters=25, seed=seed)

    # --- tokenization & histogram ---
    def _tokens(self, img_ds):
        p = extract_patches(img_ds, patch=self.patch)
        pn, mu, sd = standardize(p)
        z = ((pn[:, None, :] - self.codebook[None, :, :])**2).sum(2).argmin(1)
        return z, mu, sd

    def token_hist(self, img_ds):
        z, mu, sd = self._tokens(img_ds)
        # drop near-black tiles from counts (simple foreground mask)
        p = extract_patches(img_ds, patch=self.patch)
        w = (p.mean(1) > self.bg_threshold).astype(np.float32)
        return np.bincount(z, weights=w, minlength=self.K).astype(np.float64)

    # --- generative learning (Dirichlet–Categorical) ---
    def fit_generative(self, images_ds, labels):
        counts_c  = np.zeros(self.C)
        counts_ck = np.zeros((self.C, self.K))
        for img, y in zip(images_ds, labels):
            h = self.token_hist(img)
            counts_c[y]  += 1.0
            counts_ck[y] += h
        self.pi  = (self.alpha + counts_c);  self.pi  /= self.pi.sum()
        self.phi = (self.beta  + counts_ck); self.phi /= self.phi.sum(1, keepdims=True)

    # --- inference ---
    def log_post(self, img_ds):
        h  = self.token_hist(img_ds)
        lp = np.log(self.pi + 1e-12) + h @ np.log(self.phi + 1e-12).T
        lp -= lp.max(); p = np.exp(lp); return np.log(p / p.sum())

    def predict(self, img_ds):
        return int(self.log_post(img_ds).argmax())

    # --- reconstruction (for visuals only) ---
    def reconstruct(self, img_ds):
        H, W = img_ds.shape; ph = pw = self.patch
        rows, cols = H // ph, W // pw
        p = extract_patches(img_ds, patch=self.patch)
        pn, mu, sd = standardize(p)
        z = ((pn[:, None, :] - self.codebook[None, :, :])**2).sum(2).argmin(1)
        cent = self.codebook[z] * sd[:, None] + mu[:, None]        # denormalize per tile
        out = np.zeros((H, W), dtype=np.float32)
        k = 0
        for i in range(rows):
            for j in range(cols):
                out[i*ph:(i+1)*ph, j*pw:(j+1)*pw] = cent[k].reshape(ph, pw)
                k += 1
        return out

    def estimate_bpp(self):
        return np.log2(self.K) / (self.patch * self.patch)

# ------------------------------
# experiment
# ------------------------------

def load_mnist(limit_train=20000, limit_test=5000, seed=0):
    X, y = fetch_openml('mnist_784', version=1, as_frame=False, return_X_y=True)
    X = X.reshape(-1, 28, 28).astype(np.float32) / 255.0
    y = y.astype(np.int64)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=10000, random_state=seed, stratify=y)
    return Xtr[:limit_train], ytr[:limit_train], Xte[:limit_test], yte[:limit_test]

def preprocess_ds(X):
    return np.stack([downsample(x, 16, 16) for x in X])

def evaluate(model, X, y):
    preds = [model.predict(x) for x in X]
    acc = (np.array(preds) == y).mean()
    nll = -np.mean([model.log_post(x)[label] for x, label in zip(X, y)])
    return acc, float(nll)

def main():
    Xtr, ytr, Xte, yte = load_mnist()
    Xtr_ds, Xte_ds = preprocess_ds(Xtr), preprocess_ds(Xte)

    model = RGMImageClassifier(num_classes=10, K=256, alpha=1.0, beta=1.0, patch=4)
    print("learning codebook…"); model.fit_codebook(Xtr_ds, seed=42)
    print("fitting generative model…"); model.fit_generative(Xtr_ds, ytr)

    acc_tr, nll_tr = evaluate(model, Xtr_ds, ytr)
    acc_te, nll_te = evaluate(model, Xte_ds, yte)
    print(f"train acc={acc_tr:.3f}  nll={nll_tr:.3f}")
    print(f"test  acc={acc_te:.3f}  nll={nll_te:.3f}")

    bpp = model.estimate_bpp()
    print(f"estimated bitrate ≈ {bpp:.3f} bits/pixel (baseline: 8.000 bpp grayscale)")

    idx = np.random.default_rng(0).integers(len(Xte_ds), size=6)
    imgs = Xte_ds[idx]
    recons = [model.reconstruct(x) for x in imgs]
    psnrs = [psnr(x, r, data_range=1.0) for x, r in zip(imgs, recons)]

    fig = plt.figure(figsize=(10, 4))
    for i in range(6):
        plt.subplot(2, 6, i + 1); plt.imshow(imgs[i], cmap='gray', vmin=0, vmax=1); plt.axis('off')
        if i == 0: plt.title('downsampled')
        plt.subplot(2, 6, 6 + i + 1); plt.imshow(recons[i], cmap='gray', vmin=0, vmax=1); plt.axis('off')
        if i == 0: plt.title('reconstructed')
    plt.suptitle(f"RGM‑lite reconstructions | K={model.K}, patch={model.patch} | PSNR≈{np.mean(psnrs):.1f} dB")
    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()
