# ============================================================
# Group information
# ============================================================

group_names     = ['Selin Yucebiyik', 'Maria Paula Sanchez', 'Antonis Adamou', 'Ara Mokree']
cid_numbers     = ['01868843', '06045575', '06067135', '06036735']
oral_assessment = [0,1]

# ============================================================
# Imports (allowed packages only)
# ============================================================

import numpy as np
import time
from scipy.spatial.distance import cdist
from scipy.stats import norm

# Coursework simulator
import MLCE_CWBO2025.virtual_lab as virtual_lab
def objective_func(X):
    return np.array(virtual_lab.conduct_experiment(X))

# ============================================================
# Reproducibility
# ============================================================

REPRODUCIBLE_SEED = None  # change this if you want a different but repeatable run
rng = np.random.default_rng(REPRODUCIBLE_SEED)

# ============================================================
# Bounds (raw units)
# ============================================================

TEMP_MIN, TEMP_MAX = 30.0, 40.0
PH_MIN,   PH_MAX   = 6.0,  8.0
F_MIN,    F_MAX    = 0.0,  50.0
CELLTYPES = ['celltype_1','celltype_2','celltype_3']

# ============================================================
# Encoding + scaling
# ============================================================

ct_map = {
    'celltype_1': np.array([1,0,0],float),
    'celltype_2': np.array([0,1,0],float),
    'celltype_3': np.array([0,0,1],float),
}

def scale(x, lo, hi): return (x-lo)/(hi-lo+1e-12)
def unscale(z, lo, hi): return lo + z*(hi-lo)

def encode_point(x):
    t,p,f1,f2,f3,ct = x
    base = [
        scale(t,TEMP_MIN,TEMP_MAX),
        scale(p,PH_MIN,PH_MAX),
        scale(f1,F_MIN,F_MAX),
        scale(f2,F_MIN,F_MAX),
        scale(f3,F_MIN,F_MAX),
    ]
    base.extend(ct_map[ct])
    return np.array(base,dtype=np.float32)

def encode_X(X):
    return np.vstack([encode_point(x) for x in X]).astype(np.float32)

# ============================================================
# Candidate pools (deterministic with rng)
# ============================================================

def clip(v, lo, hi): return max(lo,min(hi,v))

def global_pool(n=6000):
    z = rng.random((n,5))  # uniform in [0,1], deterministic via rng
    out=[]
    for i in range(n):
        t = unscale(z[i,0],TEMP_MIN,TEMP_MAX)
        p = unscale(z[i,1],PH_MIN,PH_MAX)
        f1= unscale(z[i,2],F_MIN,F_MAX)
        f2= unscale(z[i,3],F_MIN,F_MAX)
        f3= unscale(z[i,4],F_MIN,F_MAX)
        ct = CELLTYPES[i % 3]  # cycle deterministically
        out.append([t,p,f1,f2,f3,ct])
    return out

def tr_pool(best_x, R, n=5000, best_ct=None, p_ct=0.9):
    (t0,p0,f10,f20,f30,_) = best_x
    out=[]
    for i in range(n):
        t  = clip(t0 + (2*rng.random()-1)*R['t'], TEMP_MIN, TEMP_MAX)
        pH = clip(p0 + (2*rng.random()-1)*R['pH'], PH_MIN,  PH_MAX)
        f1 = clip(f10 + (2*rng.random()-1)*R['f1'], F_MIN,   F_MAX)
        f2 = clip(f20 + (2*rng.random()-1)*R['f2'], F_MIN,   F_MAX)
        f3 = clip(f30 + (2*rng.random()-1)*R['f3'], F_MIN,   F_MAX)
        # deterministic CT choice with priority
        if (best_ct is not None) and (rng.random() < p_ct):
            ct = best_ct
        else:
            ct = CELLTYPES[i % 3]  # cycle deterministically
        out.append([t,pH,f1,f2,f3,ct])
    return out

# ============================================================
# GP surrogate (ARD RBF kernel)
# ============================================================

def ard_kernel(X1,X2,ls,var):
    X1=X1/(ls+1e-12); X2=X2/(ls+1e-12)
    D=cdist(X1,X2,'sqeuclidean')
    return var*np.exp(-0.5*D)

class GP:
    def __init__(self,noise=1e-6): self.noise=noise

    def fit(self,X,Y):
        self.X=np.atleast_2d(X)
        Y=np.array(Y,float)

        self.mean=float(np.mean(Y))
        self.std=float(np.std(Y)) if np.std(Y)>1e-12 else 1.0
        Ystd=(Y-self.mean)/self.std

        # heuristic ARD lengthscales (deterministic)
        ls=[]
        for j in range(self.X.shape[1]):
            col=self.X[:,j][:,None]
            D=cdist(col,col)
            md=np.median(D[D>0]) if np.any(D>0) else 0.3
            ls.append(max(md,0.1))
        self.ls=np.array(ls,float)
        self.var=max(np.var(Ystd),1.0)

        K=ard_kernel(self.X,self.X,self.ls,self.var)
        K+=self.noise*np.eye(len(self.X))
        self.L=np.linalg.cholesky(K)
        self.alpha=np.linalg.solve(self.L.T,np.linalg.solve(self.L,Ystd))

    def predict(self,Xs):
        Xs=np.atleast_2d(Xs)
        Kc=ard_kernel(Xs,self.X,self.ls,self.var)
        mu_std=Kc.dot(self.alpha)
        v=np.linalg.solve(self.L,Kc.T)
        var=np.maximum( np.diag(ard_kernel(Xs,Xs,self.ls,self.var)) - np.sum(v*v,axis=0), 1e-12 )
        mu = mu_std*self.std + self.mean
        return mu, np.sqrt(var)

# ============================================================
# Expected Improvement (deterministic)
# ============================================================

def EI(mu,sd,best_y,xi):
    sd=np.maximum(sd,1e-12)
    imp=mu-best_y-xi
    z=imp/sd
    return imp*norm.cdf(z) + sd*norm.pdf(z)

# ============================================================
# Batch EI selector (Kriging-Believer without jitter => deterministic)
# ============================================================

class BatchSelector:
    def __init__(self,pool,Xobs,Yobs,batch):
        self.pool=pool
        self.batch=batch
        self.Xobs=list(Xobs)
        self.Yobs=np.array(Yobs,float)

        self.P=encode_X(pool)
        self.mask=np.ones(len(pool),bool)

        seen=set(tuple(x) for x in Xobs)
        for i,x in enumerate(pool):
            if tuple(x) in seen:
                self.mask[i]=False

    def select(self,xi):
        chosen=[]
        gp=GP()
        Xobs_enc = encode_X(self.Xobs)
        gp.fit(Xobs_enc, self.Yobs)

        M = 1200
        K = 200

        for _ in range(self.batch):
            idx=np.where(self.mask)[0]
            if len(idx)==0: 
                break
            if len(idx)>M:
                sub_idx = idx [:M]
            else:
                sub_idx = idx

            mu_s,sd_s=gp.predict(self.P[sub_idx])
            best_y=np.max(self.Yobs)
            ei_s=EI(mu_s,sd_s,best_y,xi)

            k = min(K, len(sub_idx))
            top_rel = np.argpartition(ei_s, -k)[-k:]
            cand_idx = sub_idx[np.argsort(ei_s[top_rel])][::-1]

            mu_r, sd_r = gp.predict(self.P[cand_idx])
            ei_r = EI(mu_r, sd_r, best_y, xi)

            pick_rel=int(np.argmax(ei_r))
            pick_abs=int(cand_idx[pick_rel])

            x=self.pool[pick_abs]
            chosen.append(x)
            self.mask[pick_abs]=False

            # Believer fantasy = posterior mean (no jitter -> reproducible)
            fantasy=float(mu_r[pick_rel])
            self.Xobs.append(x)
            self.Yobs=np.concatenate([self.Yobs,[fantasy]])

            Xobs_enc = encode_X(self.Xobs)
            gp.fit(Xobs_enc,self.Yobs)

        return chosen

# ============================================================
# Main BO with timing fixed & deterministic sampling
# ============================================================

class BO:
    def __init__(self, X0, iters, batch):
        self.X = list(X0)

        # timing arrays aligned with Y (one nonzero per evaluated batch)
        self.iter_time_ms = []   # total per iteration (pool + GP + EI + simulator)
        self.eval_time_ms = []   # simulator-only per batch

        # initial evaluation (timed correctly)
        t0_iter = time.time()
        t0_eval = time.time()
        self.Y = objective_func(self.X).astype(float)
        dt_eval_ms = (time.time() - t0_eval) * 1000.0
        dt_iter_ms = (time.time() - t0_iter) * 1000.0
        self.eval_time_ms = [dt_eval_ms] + [0.0] * (len(self.Y) - 1)
        self.iter_time_ms = [dt_iter_ms] + [0.0] * (len(self.Y) - 1)

        # best trackers
        self.best_y = float(np.max(self.Y))
        self.best_x = self.X[int(np.argmax(self.Y))]
        self.best_ct = self.best_x[5]

        # trust region setup
        self.R    = {'t':5, 'pH':1, 'f1':18, 'f2':18, 'f3':18}
        self.Rmin = {'t':1.5,'pH':0.4,'f1':6,  'f2':6,  'f3':6}
        self.grow, self.shrink = 1.25, 0.75
        self.no_improve = 0

        for i in range(iters):
            iter_start = time.time()
            xi = 0.15 * (0.6 ** i)

            # pool logic (deterministic; early diversity; hard restarts)
            if i < 5:
                pool = global_pool(6000)
                for j in range(len(pool)):
                    pool[j][5] = CELLTYPES[j % 3]  # force equal CTs early
            elif self.no_improve >= 2:
                pool = global_pool(8000)
                self.R = {'t':5, 'pH':1, 'f1':18, 'f2':18, 'f3':18}
                self.no_improve = 0
            else:
                pool = tr_pool(self.best_x, self.R, 5000, best_ct=self.best_ct, p_ct=0.8)

            selector = BatchSelector(pool, self.X, self.Y, batch)

            # acquisition (NOT counted in eval_time_ms)
            Xnew = selector.select(xi)

            # simulator-only timing starts here
            eval_start = time.time()
            Ynew = objective_func(Xnew).astype(float)
            eval_dt_ms = (time.time() - eval_start) * 1000.0

            # update data
            self.X.extend(Xnew)
            self.Y = np.concatenate([self.Y, Ynew])

            # best + TR update
            old_best = self.best_y
            self.best_y = float(np.max(self.Y))
            self.best_x = self.X[int(np.argmax(self.Y))]
            self.best_ct = self.best_x[5]
            if self.best_y > old_best + 1e-9:
                for k in self.R: self.R[k] = min(self.R[k] * self.grow, self.Rmin[k] * 4)
                self.no_improve = 0
            else:
                for k in self.R: self.R[k] = max(self.R[k] * self.shrink, self.Rmin[k])
                self.no_improve += 1

            # print best titre after this iteration
            print(f"[Iteration {i+1}] Best titre so far: {self.best_y:.4f} g/L")

            # total iteration time
            iter_dt_ms = (time.time() - iter_start) * 1000.0

            # align timing with Y (one nonzero per evaluated batch)
            self.eval_time_ms += [eval_dt_ms] + [0.0] * (len(Ynew) - 1)
            self.iter_time_ms += [iter_dt_ms] + [0.0] * (len(Ynew) - 1)

    # helper for plotting/reporting
    def progress_series(self):
        """
        Returns:
          cum_iter_ms: cumulative total iteration time (ms)
          cum_eval_ms: cumulative simulator-only time (ms)
          best_so_far: running max of Y
          all_Y:       all observed Y
        """
        y = np.asarray(self.Y, float)
        t_iter = np.asarray(self.iter_time_ms, float)
        t_eval = np.asarray(self.eval_time_ms, float)
        return np.cumsum(t_iter), np.cumsum(t_eval), np.maximum.accumulate(y), y

# ============================================================
# Execution
# ============================================================

# We need to explore how best to build the search space and how to best choose initial training points! (randomly? regular-sampling? uniform/non-uniform?)
X_initial= ([[33, 6.25, 10, 20, 20, 'celltype_1'],
              [38, 8, 20, 10, 20, 'celltype_3'],
              [37, 6.8, 0, 50, 0, 'celltype_1'],
              [36, 6.0, 20, 20, 10, 'celltype_2'],
              [31, 7.50, 30, 10, 10,'celltype_3'],
              [34, 7.00, 15, 25, 10,'celltype_2']])

if __name__=="__main__":
    start_time = time.time() # <--- Timer Started

    print(">>> Running Prioritised Trust-Region BO...")
    bo=BO(X_initial,iters=15,batch=5)
    
    end_time = time.time()   # <--- Timer Stopped
    
    print(">>> Best observed titre:",bo.best_y)
    print(">>> Best conditions:",bo.best_x)
    print(f">>> Total execution time: {end_time - start_time:.2f} seconds")

    # ============================================================
    # Enhanced plotting
    # ============================================================
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    Y = np.array(bo.Y)
    # Use simulator-only timing per batch (aligned with Y)
    times = np.array(bo.eval_time_ms)
    iters = np.arange(len(Y))
    cum_Y = np.cumsum(Y)
    cum_time = np.cumsum(times)

    plt.style.use("seaborn-v0_8")
    fig, axs = plt.subplots(5, 1, figsize=(12, 22))
    fig.tight_layout(pad=4)

    # 1. Time per iteration (simulator)
    max_time = np.max(times) + 1e-9
    colors = cm.viridis(times / max_time)
    axs[0].scatter(iters, times, c=colors, s=40, edgecolor='black')
    axs[0].set_title("Time per Batch Evaluation", fontsize=14, fontweight="bold")
    axs[0].set_xlabel("Evaluation Index")
    axs[0].set_ylabel("Time [ms]")
    axs[0].grid(True, linestyle='--', alpha=0.5)

    # 2. Titre concentration per evaluation
    axs[1].plot(iters, Y, linewidth=2)
    axs[1].fill_between(iters, Y, alpha=0.3)
    axs[1].set_title("Titre Concentration per Evaluation", fontsize=14, fontweight="bold")
    axs[1].set_xlabel("Evaluation Index")
    axs[1].set_ylabel("Titre [g/L]")
    axs[1].grid(True, linestyle='--', alpha=0.5)

    # 3. Cumulative time
    axs[2].plot(iters, cum_time, linewidth=2)
    axs[2].fill_between(iters, cum_time, alpha=0.3)
    axs[2].set_title("Cumulative Time", fontsize=14, fontweight="bold")
    axs[2].set_xlabel("Evaluation Index")
    axs[2].set_ylabel("Cumulative Time [ms]")
    axs[2].grid(True, linestyle='--', alpha=0.5)

    # 4. Cumulative Titre
    axs[3].plot(iters, cum_Y, linewidth=2)
    axs[3].fill_between(iters, cum_Y, alpha=0.2)
    axs[3].set_title("Cumulative Titre", fontsize=14, fontweight="bold")
    axs[3].set_xlabel("Evaluation Index")
    axs[3].set_ylabel("Cumulative Titre [g/L]")
    axs[3].grid(True, linestyle='--', alpha=0.5)

    # 5. Cumulative Titre vs Cumulative Time
    axs[4].plot(cum_time, cum_Y, linewidth=3)
    axs[4].scatter(cum_time, cum_Y, s=20)
    axs[4].set_title("Cumulative Titre vs Cumulative Time", fontsize=14, fontweight="bold")
    axs[4].set_xlabel("Cumulative Time [ms]")
    axs[4].set_ylabel("Cumulative Titre [g/L]")
    axs[4].grid(True, linestyle='--', alpha=0.5)

    plt.show()