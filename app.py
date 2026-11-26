"""
Random-subgraph consensus explorer
==================================

Graph G (n nodes)  ->  at every time step k pick m = n-2 random nodes
  G_k = subgraph of G induced by the picked nodes   (k = 0 is random too)
        The m nodes are re-drawn until every picked node has at least one
        edge inside G_k, so L_k never has an all-zero row.
  L_k = Laplacian of G_k, only the picked nodes -> m x m (e.g. 4 x 4)
  W_k = I - eps * L_k                                  (m x m)
  tau(W_k)   = Dobrushin coefficient = 1/2 * max_{i,j} sum_l |w_il - w_jl|
  alpha(W_k) = min_{i,j} (row_i . row_j)

After K steps:
  prod_k tau(W_k),  prod_k alpha(W_k)
  P = W_K ... W_1 W_0  ->  tau(P), alpha(P)

Requirements:  pip install numpy matplotlib      (networkx optional)
Run:           py app.py
"""

import csv
import re

import numpy as np

# --------------------------------------------------------------------------
#  Math core (no GUI needed)
# --------------------------------------------------------------------------

DEFAULT_N = 6
DEFAULT_EDGES = "1-2, 2-3, 2-4, 3-4, 4-5, 5-6"
# positions that mimic the whiteboard drawing
DEFAULT_POS = {0: (0.0, 1.0), 1: (1.5, 0.8), 2: (3.0, 0.3),
               3: (1.2, -0.3), 4: (2.5, -0.9), 5: (4.0, -0.9)}


def parse_edges(text, n):
    """'1-2, 2-3' or '1 2\\n2 3' -> list of 0-based (u, v) pairs."""
    edges = set()
    for token in re.split(r"[,;\n]+", text):
        nums = re.findall(r"\d+", token)
        if not nums:
            continue
        if len(nums) != 2:
            raise ValueError(f"Can't read edge '{token.strip()}' (need two node numbers)")
        u, v = int(nums[0]) - 1, int(nums[1]) - 1
        if not (0 <= u < n and 0 <= v < n):
            raise ValueError(f"Edge '{token.strip()}' uses a node outside 1..{n}")
        if u == v:
            raise ValueError(f"Self-loop '{token.strip()}' not allowed")
        edges.add((min(u, v), max(u, v)))
    return sorted(edges)


def laplacian(n, edges, active=None):
    """n x n Laplacian. If `active` is given, only edges with both ends active count."""
    L = np.zeros((n, n))
    act = None if active is None else set(active)
    for u, v in edges:
        if act is None or (u in act and v in act):
            L[u, u] += 1
            L[v, v] += 1
            L[u, v] -= 1
            L[v, u] -= 1
    return L


def tau(W):
    """Dobrushin ergodicity coefficient: 1/2 max_{i,j} ||w_i - w_j||_1
    (= 1 - min_{i,j} sum_l min(w_il, w_jl) for stochastic W)."""
    if W.shape[0] < 2:
        return 0.0
    D = 0.5 * np.abs(W[:, None, :] - W[None, :, :]).sum(axis=2)
    return float(D.max())


def alpha(W, include_diagonal=True):
    """min over pairs of rows of the dot product row_i . row_j."""
    if W.shape[0] == 0:
        return float("nan")
    G = W @ W.T
    if not include_diagonal and W.shape[0] > 1:
        G = G[~np.eye(W.shape[0], dtype=bool)]
    return float(G.min())


def pick_nodes(rng, n, edges, m, max_tries=20000):
    """Pick m random nodes so that every picked node has at least one edge
    to another picked node (no all-zero row in L_k)."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    for _ in range(max_tries):
        active = sorted(int(x) for x in rng.choice(n, size=m, replace=False))
        chosen = set(active)
        if all(adj[v] & chosen for v in active):
            return active
    raise ValueError(f"Couldn't find {m} nodes where every node has an edge to "
                     f"another picked node. Try a different number of nodes per step.")


def run_algorithm(n, edges, eps, K, m, seed=None, include_diagonal=True,
                  product_full=False):
    """Run k = 0..K. Every step (k = 0 included) picks m random nodes with no
    isolated node, so L_k and W_k are always m x m with no zero row.
    product_full=False: P = product of the m x m matrices as they are.
    product_full=True : each W_k is put back at its own nodes inside an
                        n x n identity matrix before multiplying."""
    rng = np.random.default_rng(seed)
    steps = []
    for k in range(K + 1):
        active = pick_nodes(rng, n, edges, m)
        L_full = laplacian(n, edges, active)
        L = L_full[np.ix_(active, active)]
        W = np.eye(m) - eps * L
        steps.append({"k": k, "active": active, "L": L, "W": W,
                      "tau": tau(W), "alpha": alpha(W, include_diagonal)})

    P = np.eye(n if product_full else m)
    for s in steps:                 # P = W_K ... W_1 W_0
        if product_full:
            Wbig = np.eye(n)
            Wbig[np.ix_(s["active"], s["active"])] = s["W"]
            P = Wbig @ P
        else:
            P = s["W"] @ P
    summary = {
        "prod_tau": float(np.prod([s["tau"] for s in steps])),
        "prod_alpha": float(np.prod([s["alpha"] for s in steps])),
        "tau_P": tau(P),
        "alpha_P": alpha(P, include_diagonal),
        "P": P,
        "product_full": product_full,
    }
    return steps, summary


def fmt_matrix(M, decimals=4, labels=None):
    n = M.shape[0]
    labels = labels or [str(i + 1) for i in range(n)]
    w = max(decimals + 4, 5)
    head = " " * 4 + "".join(f"{labels[j]:>{w}}" for j in range(n))
    rows = [head]
    for i in range(n):
        vals = []
        for x in M[i]:
            x = 0.0 if abs(x) < 10 ** (-decimals - 1) else x
            vals.append(f"{x:>{w}.{decimals}f}")
        rows.append(f"{labels[i]:>3} " + "".join(vals))
    return "\n".join(rows)


def fmt_num(x):
    if x != x:
        return "undefined (no nodes left)"
    return f"{x:.6g}" if x != 0 else "0"


def graph_positions(n, edges):
    if n == DEFAULT_N and edges == parse_edges(DEFAULT_EDGES, DEFAULT_N):
        return DEFAULT_POS
    try:
        import networkx as nx
        g = nx.Graph()
        g.add_nodes_from(range(n))
        g.add_edges_from(edges)
        return {k: tuple(v) for k, v in nx.spring_layout(g, seed=1).items()}
    except ImportError:
        ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return {i: (np.cos(a), np.sin(a)) for i, a in enumerate(ang)}


# --------------------------------------------------------------------------
#  GUI
# --------------------------------------------------------------------------

try:
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog
except ImportError:          # allows importing the math core without tkinter
    tk = None


def draw_graph(ax, n, edges, pos, active=None, title=""):
    ax.clear()
    act = set(range(n)) if active is None else set(active)
    for u, v in edges:
        on = u in act and v in act
        (x1, y1), (x2, y2) = pos[u], pos[v]
        ax.plot([x1, x2], [y1, y2],
                color="#1f4e8c" if on else "#cccccc",
                lw=2.6 if on else 1.2, ls="-" if on else "--", zorder=1)
    for i in range(n):
        x, y = pos[i]
        on = i in act
        ax.scatter([x], [y], s=900, zorder=2,
                   color="#f4a742" if on else "white",
                   edgecolors="#333333" if on else "#bbbbbb", linewidths=1.8)
        ax.text(x, y, str(i + 1), ha="center", va="center", fontsize=13,
                color="black" if on else "#aaaaaa", zorder=3, fontweight="bold")
    ax.set_title(title, fontsize=12)
    ax.set_aspect("equal")
    ax.margins(0.18)
    ax.axis("off")


class App:
    def __init__(self, root):
        import matplotlib
        matplotlib.use("TkAgg")
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        self.Figure, self.Canvas = Figure, FigureCanvasTkAgg

        self.root = root
        root.title("Random subgraph consensus — L_k, W_k, τ, α")
        root.geometry("1250x800")
        self.steps, self.summary = None, None

        # ---------------- left: inputs ----------------
        left = ttk.Frame(root, padding=10)
        left.pack(side="left", fill="y")

        ttk.Label(left, text="Graph G", font=("", 12, "bold")).pack(anchor="w")
        self.n_var = tk.StringVar(value=str(DEFAULT_N))
        self._row(left, "Number of nodes n", self.n_var)
        ttk.Label(left, text="Edges (e.g. 1-2, 2-3):").pack(anchor="w", pady=(6, 0))
        self.edges_txt = tk.Text(left, width=30, height=5, wrap="word")
        self.edges_txt.insert("1.0", DEFAULT_EDGES)
        self.edges_txt.pack(anchor="w")
        ttk.Button(left, text="Show G, L, W", command=self.show_original).pack(fill="x", pady=4)

        ttk.Separator(left).pack(fill="x", pady=8)
        ttk.Label(left, text="Algorithm", font=("", 12, "bold")).pack(anchor="w")
        self.eps_var = tk.StringVar(value="0.2")
        self.K_var = tk.StringVar(value="50")
        self.m_var = tk.StringVar(value=str(DEFAULT_N - 2))
        self.seed_var = tk.StringVar(value="")
        self._row(left, "ε (epsilon)", self.eps_var)
        self._row(left, "Final step K (you choose)", self.K_var)
        self._row(left, "Nodes picked per step", self.m_var)
        self._row(left, "Random seed (blank = new)", self.seed_var)
        self.n_var.trace_add("write", self._n_changed)

        self.full_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(left, text="Π W_k: put each W_k back to n x n",
                        variable=self.full_var).pack(anchor="w", pady=2)
        self.diag_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(left, text="α: include i = j in the min",
                        variable=self.diag_var).pack(anchor="w", pady=2)

        ttk.Button(left, text="▶  RUN", command=self.run).pack(fill="x", pady=(10, 4))
        ttk.Button(left, text="Export results to CSV", command=self.export).pack(fill="x")
        self.status = ttk.Label(left, text="", foreground="#a33", wraplength=230)
        self.status.pack(anchor="w", pady=8)

        # ---------------- right: tabs ----------------
        self.nb = ttk.Notebook(root)
        self.nb.pack(side="left", fill="both", expand=True)

        # tab 1: original graph
        t1 = ttk.Frame(self.nb)
        self.nb.add(t1, text="Original G")
        self.fig0, self.canvas0, self.text0 = self._graph_and_text(t1)

        # tab 2: step by step
        t2 = ttk.Frame(self.nb)
        self.nb.add(t2, text="Step by step (G_k)")
        nav = ttk.Frame(t2, padding=4)
        nav.pack(fill="x")
        ttk.Button(nav, text="◀ prev", command=lambda: self.goto(self.k_var.get() - 1)).pack(side="left")
        self.k_var = tk.IntVar(value=0)
        self.k_scale = tk.Scale(nav, from_=0, to=0, orient="horizontal", variable=self.k_var,
                                command=lambda _v: self.show_step(), length=500, label="k")
        self.k_scale.pack(side="left", padx=8)
        ttk.Button(nav, text="next ▶", command=lambda: self.goto(self.k_var.get() + 1)).pack(side="left")
        self.fig1, self.canvas1, self.text1 = self._graph_and_text(t2)

        # tab 3: summary
        t3 = ttk.Frame(self.nb)
        self.nb.add(t3, text="Results (products)")
        top = ttk.Frame(t3)
        top.pack(fill="both", expand=True)
        cols = ("k", "nodes", "tau", "alpha")
        self.tree = ttk.Treeview(top, columns=cols, show="headings", height=14)
        for c, wdt, lab in zip(cols, (50, 140, 150, 150), ("k", "nodes picked", "τ(W_k)", "α(W_k)")):
            self.tree.heading(c, text=lab)
            self.tree.column(c, width=wdt, anchor="center")
        sb = ttk.Scrollbar(top, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side="left", fill="y", padx=(6, 0), pady=6)
        sb.pack(side="left", fill="y", pady=6)
        self.tree.bind("<Double-1>", self._tree_click)
        self.fig2 = self.Figure(figsize=(5.5, 3.6), dpi=100)
        self.canvas2 = self.Canvas(self.fig2, master=top)
        self.canvas2.get_tk_widget().pack(side="left", fill="both", expand=True)
        self.text2 = tk.Text(t3, height=16, font=("Courier", 11), wrap="none")
        self.text2.pack(fill="both", expand=True, padx=6, pady=6)

        self.show_original()

    # ---------- helpers ----------
    def _row(self, parent, label, var):
        f = ttk.Frame(parent)
        f.pack(fill="x", pady=2)
        ttk.Label(f, text=label, width=24).pack(side="left")
        ttk.Entry(f, textvariable=var, width=10).pack(side="left")

    def _graph_and_text(self, parent):
        body = ttk.Frame(parent)
        body.pack(fill="both", expand=True)
        fig = self.Figure(figsize=(5, 4), dpi=100)
        canvas = self.Canvas(fig, master=body)
        canvas.get_tk_widget().pack(side="left", fill="both", expand=True)
        txt = tk.Text(body, width=62, font=("Courier", 11), wrap="none")
        txt.pack(side="left", fill="both", expand=True, padx=6, pady=6)
        return fig, canvas, txt

    def _set_text(self, widget, s):
        widget.config(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", s)
        widget.config(state="disabled")

    def _n_changed(self, *_):
        try:
            self.m_var.set(str(max(1, int(self.n_var.get()) - 2)))
        except ValueError:
            pass

    def _read_graph(self):
        n = int(self.n_var.get())
        if n < 2:
            raise ValueError("n must be at least 2")
        edges = parse_edges(self.edges_txt.get("1.0", "end"), n)
        return n, edges

    def _eps_warning(self, n, edges, eps):
        dmax = laplacian(n, edges).diagonal().max()
        if dmax > 0 and eps > 1 / dmax:
            return (f"Note: ε = {eps} > 1/max degree = {1 / dmax:.4g}, "
                    "so W can have negative entries (not stochastic).")
        return ""

    # ---------- actions ----------
    def show_original(self):
        try:
            n, edges = self._read_graph()
            eps = float(self.eps_var.get())
        except ValueError as e:
            messagebox.showerror("Input error", str(e))
            return
        self.pos = graph_positions(n, edges)
        ax = self.fig0.gca() if self.fig0.axes else self.fig0.add_subplot(111)
        draw_graph(ax, n, edges, self.pos, title="G (original graph)")
        self.canvas0.draw()
        L = laplacian(n, edges)
        W = np.eye(n) - eps * L
        s = (f"Edges: {', '.join(f'{u + 1}-{v + 1}' for u, v in edges)}\n\n"
             f"L =\n{fmt_matrix(L, 0)}\n\n"
             f"W = I - ε L   (ε = {eps})\n{fmt_matrix(W)}\n\n"
             f"τ(W) = {fmt_num(tau(W))}\n"
             f"α(W) = {fmt_num(alpha(W, self.diag_var.get()))}\n")
        self._set_text(self.text0, s)
        self.status.config(text=self._eps_warning(n, edges, eps))

    def run(self):
        try:
            n, edges = self._read_graph()
            eps = float(self.eps_var.get())
            K = int(self.K_var.get())
            m = int(self.m_var.get())
            seed = int(self.seed_var.get()) if self.seed_var.get().strip() else None
            if K < 0:
                raise ValueError("K must be ≥ 0")
            if not 1 <= m <= n:
                raise ValueError(f"Nodes per step must be between 1 and n = {n}")
        except ValueError as e:
            messagebox.showerror("Input error", str(e))
            return

        self.n, self.edges, self.eps = n, edges, eps
        self.pos = graph_positions(n, edges)
        try:
            self.steps, self.summary = run_algorithm(
                n, edges, eps, K, m, seed, self.diag_var.get(), self.full_var.get())
        except ValueError as e:
            messagebox.showerror("Can't pick nodes", str(e))
            return
        self.status.config(text=self._eps_warning(n, edges, eps))

        self.k_scale.config(to=K)
        self.k_var.set(0)
        self.show_original()
        self.show_step()
        self.fill_summary()
        self.nb.select(1)

    def goto(self, k):
        if self.steps:
            self.k_var.set(max(0, min(k, len(self.steps) - 1)))
            self.show_step()

    def show_step(self):
        if not self.steps:
            return
        s = self.steps[self.k_var.get()]
        nodes = ", ".join(str(i + 1) for i in s["active"])
        ax = self.fig1.gca() if self.fig1.axes else self.fig1.add_subplot(111)
        draw_graph(ax, self.n, self.edges, self.pos, s["active"],
                   title=f"G_{s['k']}  (nodes {nodes})")
        self.canvas1.draw()
        lab = [str(i + 1) for i in s["active"]]
        size = len(s["active"])
        sub_edges = [f"{u + 1}-{v + 1}" for u, v in self.edges
                     if u in s["active"] and v in s["active"]]
        txt = (f"k = {s['k']}\nNodes picked: {nodes}\n"
               f"Edges of G_{s['k']}: {', '.join(sub_edges) or '(none)'}\n"
               f"\n"
               f"L_{s['k']}  ({size} x {size}, rows/cols = nodes {nodes}) =\n"
               f"{fmt_matrix(s['L'], 0, lab)}\n\n"
               f"W_{s['k']} = I - ε L_{s['k']}   (ε = {self.eps})\n{fmt_matrix(s['W'], 4, lab)}\n\n"
               f"τ(W_{s['k']}) = {fmt_num(s['tau'])}\n"
               f"α(W_{s['k']}) = {fmt_num(s['alpha'])}\n")
        self._set_text(self.text1, txt)

    def fill_summary(self):
        self.tree.delete(*self.tree.get_children())
        for s in self.steps:
            self.tree.insert("", "end", values=(
                s["k"], ",".join(str(i + 1) for i in s["active"]),
                fmt_num(s["tau"]), fmt_num(s["alpha"])))

        ks = [s["k"] for s in self.steps]
        self.fig2.clear()
        ax = self.fig2.add_subplot(111)
        ax.plot(ks, [s["tau"] for s in self.steps], "o-", ms=3, label="τ(W_k)")
        ax.plot(ks, [s["alpha"] for s in self.steps], "s-", ms=3, label="α(W_k)")
        ax.set_xlabel("k")
        ax.legend()
        ax.grid(alpha=0.3)
        self.fig2.tight_layout()
        self.canvas2.draw()

        S, K = self.summary, self.steps[-1]["k"]
        if S["product_full"]:
            where = "each W_k placed at its own nodes, identity elsewhere"
            check = ""
        else:
            where = "rows/cols = positions 1..m of each step"
            ok = "✓" if S["tau_P"] <= S["prod_tau"] + 1e-12 else "✗"
            check = f"  sanity check τ(Π W_k) ≤ Π τ(W_k):  {ok}\n\n"
        txt = (f"Steps k = 0 … {K}  ({len(self.steps)} matrices),  P = W_{K} ··· W_1 W_0\n\n"
               f"  Π τ(W_k)   = {fmt_num(S['prod_tau'])}\n"
               f"  Π α(W_k)   = {fmt_num(S['prod_alpha'])}\n\n"
               f"  τ(Π W_k)   = {fmt_num(S['tau_P'])}\n"
               f"  α(Π W_k)   = {fmt_num(S['alpha_P'])}\n\n"
               f"{check}"
               f"P = Π W_k  ({S['P'].shape[0]} x {S['P'].shape[0]}, {where}) =\n"
               f"{fmt_matrix(S['P'], 6)}\n")
        self._set_text(self.text2, txt)

    def _tree_click(self, _e):
        sel = self.tree.selection()
        if sel:
            self.goto(int(self.tree.item(sel[0], "values")[0]))
            self.nb.select(1)

    def export(self):
        if not self.steps:
            messagebox.showinfo("Nothing yet", "Press RUN first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv",
                                            filetypes=[("CSV", "*.csv")])
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["k", "nodes", "tau(W_k)", "alpha(W_k)"])
            for s in self.steps:
                w.writerow([s["k"], " ".join(str(i + 1) for i in s["active"]),
                            s["tau"], s["alpha"]])
            S = self.summary
            w.writerow([])
            w.writerow(["prod tau(W_k)", S["prod_tau"]])
            w.writerow(["prod alpha(W_k)", S["prod_alpha"]])
            w.writerow(["tau(prod W_k)", S["tau_P"]])
            w.writerow(["alpha(prod W_k)", S["alpha_P"]])
        messagebox.showinfo("Saved", f"Results saved to\n{path}")


def main():
    if tk is None:
        raise SystemExit("tkinter is not available in this Python install.")
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
