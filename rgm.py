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
    H, W = img.shape
    ph = pw = patch
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
    mu = p.mean(1, keepdims=True)
    sd = p.std(1, keepdims=True) + 1e-6
    return (p - mu) / sd, mu.squeeze(1), sd.squeeze(1)

# ------------------------------
# model: RGM‑lite (2 levels)
# ------------------------------

class RGMImageClassifier:
    """
    Backward-compatible API with a 2-scale hierarchy inside:
      Level 0 (tokens):    4x4 pixel patches -> token ids via codebook C1 (size K)
      Level 1 (supertoks): 2x2 neighborhoods of tokens -> supertoken ids via codebook C2 (size K2)
    """
    def __init__(self, num_classes=10, K=256, K2=64, alpha=1.0, beta=1.0, beta2=1.0,
                 patch=4, bg_threshold=0.10, seed=42):
        self.C = int(num_classes)
        self.K = int(K)          # token vocabulary size
        self.K2 = int(K2)        # supertoken vocabulary size (set to 0 to disable 2nd scale)
        self.alpha = float(alpha)
        self.beta = float(beta)   # smoothing for p(token | supertoken) or p(token | class) in 1-scale mode
        self.beta2 = float(beta2) # smoothing for p(supertoken | class)
        self.patch = int(patch)
        self.bg_threshold = float(bg_threshold)
        self.seed = int(seed)

        # learned parameters
        self.pi = np.full(self.C, 1.0/self.C)                 # p(class)
        self.theta = None  # (C, K2): p(supertok | class)
        self.psi   = None  # (K2, K): p(token | supertok)
        self.phi   = None  # (C, K)  : p(token | class) — used only if K2 == 0

        # codebooks
        self.C1 = None  # (K, 16)  tokens on 4x4 patches
        self.C2 = None  # (K2, 64) supertokens on 2x2 neighborhoods (concat of four 4x4 centroids)

    # ---------- public API ----------
    def fit_codebook(self, images_ds, seed=None):
        """Learn both codebooks (tokens + supertokens)."""
        if seed is None: seed = self.seed
        self._fit_codebook_tokens(images_ds, seed)
        if self.K2 > 0:
            self._fit_codebook_supertokens(images_ds, seed)

    def fit_generative(self, images_ds, labels):
        """Learn priors/conditionals. Uses 2-scale model if K2>0, else 1-scale BoVW."""
        counts_c = np.zeros(self.C)
        self.pi[:] = 1.0 / self.C

        if self.K2 > 0:
            # two-scale: p(s|c) and p(z|s)
            counts_theta = np.zeros((self.C, self.K2))
            counts_psi   = np.zeros((self.K2, self.K))
            for x, y in zip(images_ds, labels):
                zgrid = self._tokens_grid(x)
                s = self._supertokens_for_grid(zgrid)
                counts_c[y] += 1.0
                # class -> supertoken
                np.add.at(counts_theta[y], s, 1.0)
                # supertoken -> tokens (per block)
                rows, cols = zgrid.shape
                k = 0
                for i in range(0, rows, 2):
                    for j in range(0, cols, 2):
                        block_tokens = zgrid[i:i+2, j:j+2].ravel()
                        si = s[k]
                        k += 1
                        np.add.at(counts_psi[si], block_tokens, 1.0)

            self.pi   = (self.alpha + counts_c)
            self.pi   /= self.pi.sum()
            self.theta = (self.beta2 + counts_theta)
            self.theta /= self.theta.sum(1, keepdims=True)
            self.psi   = (self.beta  + counts_psi)
            self.psi   /= self.psi.sum(1, keepdims=True)
            self.phi   = None  # not used in 2-scale
        else:
            # one-scale fallback: BoVW p(z|c)
            counts_phi = np.zeros((self.C, self.K))
            for x, y in zip(images_ds, labels):
                h = self._token_hist(x)
                counts_c[y] += 1.0
                counts_phi[y] += h
            self.pi  = (self.alpha + counts_c)
            self.pi  /= self.pi.sum()
            self.phi = (self.beta  + counts_phi)
            self.phi /= self.phi.sum(1, keepdims=True)
            self.theta = self.psi = None

    def log_post(self, img_ds):
        """Class log-posterior for a single image."""
        if self.K2 > 0:
            zgrid = self._tokens_grid(img_ds)
            rows, cols = zgrid.shape

            log_theta = np.log(self.theta + 1e-12)   # (C, K2)
            log_psi   = np.log(self.psi   + 1e-12)   # (K2, K)

            # marginalize over supertokens per block:
            # log p(block | c) = logsumexp_s [ log p(s|c) + sum_z log p(z|s) ]
            lp = np.log(self.pi + 1e-12)
            for i in range(0, rows, 2):
                for j in range(0, cols, 2):
                    block_tokens = zgrid[i:i+2, j:j+2].ravel()
                    ll_tok = np.sum(log_psi[:, block_tokens], axis=1)       # (K2,)
                    log_joint = log_theta + ll_tok[None, :]                 # (C, K2)
                    mx = log_joint.max(axis=1, keepdims=True)
                    lp += mx.squeeze(1) + np.log(np.sum(np.exp(log_joint - mx), axis=1))

            lp -= lp.max()
            p = np.exp(lp)
            return np.log(p / p.sum())
        else:
            # one-scale: bag of tokens with p(z|c)
            h = self._token_hist(img_ds)
            lp = np.log(self.pi + 1e-12) + h @ np.log(self.phi + 1e-12).T
            lp -= lp.max()
            p = np.exp(lp)
            return np.log(p / p.sum())

    def predict(self, img_ds):
        return int(self.log_post(img_ds).argmax())

    def reconstruct(self, img_ds):
        """
        Visual reconstruction using the token codebook only (same look as before).
        """
        H, W = img_ds.shape
        ph = pw = self.patch
        rows, cols = H // ph, W // pw
        p = extract_patches(img_ds, patch=self.patch)
        pn, mu, sd = standardize(p)
        z = ((pn[:, None, :] - self.C1[None, :, :])**2).sum(2).argmin(1)
        cent = self.C1[z] * sd[:, None] + mu[:, None]
        out = np.zeros((H, W), dtype=np.float32)
        k = 0
        for i in range(rows):
            for j in range(cols):
                out[i*ph:(i+1)*ph, j*pw:(j+1)*pw] = cent[k].reshape(ph, pw)
                k += 1
        return out

    def estimate_bpp(self):
        """
        Bitrate estimate (no entropy coding), per pixel:
          - Level 0 tokens:     log2(K) bits per 4x4 patch => log2(K)/patch^2
          - Level 1 supertokens: log2(K2) bits per 2x2 token block (covers (2*patch)^2 pixels)
                                  => log2(K2) / ( (2*patch)^2 )
        """
        bpp_tokens = np.log2(self.K)  / (self.patch * self.patch)
        if self.K2 > 0:
            bpp_super  = np.log2(self.K2) / ( (2*self.patch) * (2*self.patch) )
            return float(bpp_tokens + bpp_super)
        return float(bpp_tokens)

    # ---------- internals (unchanged ideas) ----------
    def _fit_codebook_tokens(self, images_ds, seed):
        fg = []
        for x in images_ds:
            p = extract_patches(x, patch=self.patch)
            m = p.mean(1)
            p = p[m > self.bg_threshold]
            if len(p): fg.append(standardize(p)[0])
        X = np.vstack(fg)
        self.C1 = kmeans(X, self.K, iters=25, seed=seed)

    def _tokens_grid(self, img_ds):
        assert self.C1 is not None, "fit_codebook() first"
        H, W = img_ds.shape
        rows, cols = H // self.patch, W // self.patch
        p = extract_patches(img_ds, patch=self.patch)
        pn, _, _ = standardize(p)
        z = ((pn[:, None, :] - self.C1[None, :, :])**2).sum(2).argmin(1)
        return z.reshape(rows, cols)

    def _block_features(self, z_grid):
        """Concatenate the 4 token centroids in each 2x2 neighborhood, then standardize."""
        rows, cols = z_grid.shape
        assert rows % 2 == 0 and cols % 2 == 0, "token grid must be even to form 2x2 blocks"
        feats = []
        for i in range(0, rows, 2):
            for j in range(0, cols, 2):
                ids4 = z_grid[i:i+2, j:j+2].ravel()         # 4 token ids
                vec  = self.C1[ids4].reshape(-1)            # 4*16 = 64 dims
                mu, sd = vec.mean(), vec.std() + 1e-6
                feats.append((vec - mu) / sd)
        return np.stack(feats, axis=0)

    def _fit_codebook_supertokens(self, images_ds, seed):
        X = []
        for x in images_ds:
            zgrid = self._tokens_grid(x)
            X.append(self._block_features(zgrid))
        X = np.vstack(X)
        self.C2 = kmeans(X, self.K2, iters=25, seed=seed)

    def _supertokens_for_grid(self, z_grid):
        feats = self._block_features(z_grid)                 # [n_blocks, 64]
        d2 = ((feats[:, None, :] - self.C2[None, :, :])**2).sum(2)
        return d2.argmin(1)                                  # [n_blocks]


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
    print("learning codebook")
    model.fit_codebook(Xtr_ds, seed=42)
    print("fitting generative model")
    model.fit_generative(Xtr_ds, ytr)

    acc_tr, nll_tr = evaluate(model, Xtr_ds, ytr)
    acc_te, nll_te = evaluate(model, Xte_ds, yte)
    print(f"train acc={acc_tr:.3f}  nll={nll_tr:.3f}")
    print(f"test  acc={acc_te:.3f}  nll={nll_te:.3f}")

    bpp = model.estimate_bpp()
    print(f"estimated bitrate ~ {bpp:.3f} bits/pixel (baseline: 8.000 bpp grayscale)")

    idx = np.random.default_rng(0).integers(len(Xte_ds), size=6)
    imgs = Xte_ds[idx]
    recons = [model.reconstruct(x) for x in imgs]
    psnrs = [psnr(x, r, data_range=1.0) for x, r in zip(imgs, recons)]

    fig = plt.figure(figsize=(10, 4))
    for i in range(6):
        plt.subplot(2, 6, i + 1)
        plt.imshow(imgs[i], cmap='gray', vmin=0, vmax=1)
        plt.axis('off')
        if i == 0: plt.title('downsampled')
        plt.subplot(2, 6, 6 + i + 1)
        plt.imshow(recons[i], cmap='gray', vmin=0, vmax=1)
        plt.axis('off')
        if i == 0: plt.title('reconstructed')
    plt.suptitle(f"RGM‑lite reconstructions | K={model.K}, patch={model.patch} | PSNR≈{np.mean(psnrs):.1f} dB")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
