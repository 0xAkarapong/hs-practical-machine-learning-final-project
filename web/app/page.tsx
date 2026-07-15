"use client";

import { useEffect, useState } from "react";

type Meta = { sections: string[]; shop_locations: string[] };

type ModelMetrics = {
  rmse_log: number;
  mae_log: number;
  r2_log: number;
  rmse_thb: number;
  mae_thb: number;
  cv?: { k?: number; r2_log_mean: number; r2_log_std: number };
};

type Metrics = {
  baseline: ModelMetrics;
  gbdt: ModelMetrics;
  xgboost: ModelMetrics;
  selected_features: string[];
};

const MODELS: { key: keyof Pick<Metrics, "baseline" | "gbdt" | "xgboost">; label: string }[] = [
  { key: "baseline", label: "Baseline" },
  { key: "gbdt", label: "GBDT" },
  { key: "xgboost", label: "XGBoost" },
];

const fmt = (n: number, d = 2) =>
  Number.isFinite(n) ? n.toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d }) : "—";

export default function Home() {
  const [meta, setMeta] = useState<Meta | null>(null);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [metaErr, setMetaErr] = useState("");
  const [metricsErr, setMetricsErr] = useState("");

  const [form, setForm] = useState({
    Section: "Acne Care",
    Name: "DHC Vitamin B-Mix วิตามินบีรวม (สำหรับ 20 วัน)",
    "Total Sold": "0",
    "Total Reviews": "0",
    "Shop Location": "Bangkok",
  });
  const [price, setPrice] = useState<number | null>(null);
  const [predicting, setPredicting] = useState(false);
  const [predictErr, setPredictErr] = useState("");

  useEffect(() => {
    fetch("/api/meta")
      .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
      .then(setMeta)
      .catch(() => setMetaErr("Could not load /meta — is the API running?"));
    fetch("/api/metrics")
      .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
      .then(setMetrics)
      .catch(() => setMetricsErr("Could not load /metrics"));
  }, []);

  async function predict(e: React.FormEvent) {
    e.preventDefault();
    setPredicting(true);
    setPredictErr("");
    setPrice(null);
    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          ...form,
          "Total Sold": Number(form["Total Sold"]) || 0,
          "Total Reviews": Number(form["Total Reviews"]) || 0,
        }),
      });
      if (!res.ok) throw new Error(`API ${res.status}`);
      const data = await res.json();
      setPrice(data.predicted_price_thb);
    } catch {
      setPredictErr("Prediction failed — is the API running and trained?");
    } finally {
      setPredicting(false);
    }
  }

  return (
    <div className="flex min-h-full flex-col bg-zinc-50 text-zinc-900 dark:bg-black dark:text-zinc-100">
      <main className="mx-auto w-full max-w-3xl flex-1 px-6 py-16">
        <h1 className="text-2xl font-semibold tracking-tight">Health &amp; Wellness Price Recommender</h1>
        <p className="mt-1 text-sm text-zinc-600 dark:text-zinc-400">
          Set the <strong>section</strong>, <strong>product name</strong>, and <strong>shop
          location</strong> — those drive the price. Units sold and reviews are optional; leave
          them at 0 for a brand-new listing that has no sales history yet.
        </p>

        {/* Predictor form */}
        <form onSubmit={predict} className="mt-8 grid gap-4 rounded-xl border border-zinc-200 p-6 dark:border-zinc-800">
          <Field label="Section">
            <Select
              value={form.Section}
              onChange={(v) => setForm({ ...form, Section: v })}
              options={meta?.sections ?? [form.Section]}
            />
          </Field>
          <Field label="Product name">
            <Input value={form.Name} onChange={(v) => setForm({ ...form, Name: v })} placeholder="Product title (Thai or English)" />
          </Field>
          <Field label="Shop location">
            <Select
              value={form["Shop Location"]}
              onChange={(v) => setForm({ ...form, "Shop Location": v })}
              options={meta?.shop_locations ?? [form["Shop Location"]]}
            />
          </Field>

          {/* Optional sales history — unknown for a new listing, defaults to 0. */}
          <div className="mt-2 border-t border-zinc-200 pt-4 dark:border-zinc-800">
            <p className="mb-3 text-xs font-medium uppercase tracking-wide text-zinc-500">
              Optional · sales history
            </p>
            <div className="grid grid-cols-2 gap-4">
              <Field label="Units sold" hint="Units already sold; 0 for a new listing">
                <Input
                  type="number"
                  min={0}
                  value={form["Total Sold"]}
                  onChange={(v) => setForm({ ...form, "Total Sold": v })}
                  placeholder="0"
                />
              </Field>
              <Field label="Total reviews" hint="Reviews so far; 0 if none yet">
                <Input
                  type="number"
                  min={0}
                  value={form["Total Reviews"]}
                  onChange={(v) => setForm({ ...form, "Total Reviews": v })}
                  placeholder="0"
                />
              </Field>
            </div>
          </div>

          <div className="flex items-center gap-4">
            <button
              type="submit"
              disabled={predicting}
              className="rounded-lg bg-zinc-900 px-5 py-2.5 text-sm font-medium text-white transition hover:bg-zinc-700 disabled:opacity-50 dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-zinc-300"
            >
              {predicting ? "Predicting…" : "Predict price"}
            </button>
            {price !== null && (
              <span className="text-lg font-semibold text-emerald-600 dark:text-emerald-400">
                {fmt(price)} THB
              </span>
            )}
            {predictErr && <span className="text-sm text-red-600">{predictErr}</span>}
          </div>
          {metaErr && <p className="text-xs text-amber-600">{metaErr}</p>}
        </form>

        {/* Metrics dashboard */}
        <h2 className="mt-12 text-lg font-semibold">Model metrics</h2>
        {metricsErr && <p className="mt-2 text-sm text-amber-600">{metricsErr}</p>}
        {metrics && (
          <div className="mt-4 grid gap-4 sm:grid-cols-3">
            {MODELS.map(({ key, label }) => {
              const m = metrics[key];
              return (
                <div
                  key={key}
                  className="rounded-xl border border-zinc-200 p-4 dark:border-zinc-800"
                >
                  <div className="font-medium">{label}</div>
                  <dl className="mt-3 space-y-1 text-sm">
                    <Row k="R² (log)" v={fmt(m.r2_log, 4)} />
                    <Row k="RMSE (THB)" v={fmt(m.rmse_thb)} />
                    <Row k="MAE (THB)" v={fmt(m.mae_thb)} />
                    {m.cv && (
                      <Row
                        k={`CV R² (k=${m.cv.k ?? 5})`}
                        v={`${fmt(m.cv.r2_log_mean, 3)} ± ${fmt(m.cv.r2_log_std, 3)}`}
                      />
                    )}
                  </dl>
                </div>
              );
            })}
          </div>
        )}
        {metrics && (
          <p className="mt-3 text-xs text-zinc-500">
            {metrics.selected_features.length} selected features feed the deployed XGBoost model.
          </p>
        )}
      </main>
    </div>
  );
}

function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <label className="grid gap-1.5 text-sm">
      <span className="font-medium">{label}</span>
      {children}
      {hint && <span className="text-xs text-zinc-500 dark:text-zinc-400">{hint}</span>}
    </label>
  );
}

function Input({
  value,
  onChange,
  placeholder,
  type = "text",
}: {
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
  type?: string;
}) {
  return (
    <input
      type={type}
      value={value}
      placeholder={placeholder}
      onChange={(e) => onChange(e.target.value)}
      className="rounded-lg border border-zinc-300 bg-white px-3 py-2 text-sm outline-none focus:border-zinc-900 dark:border-zinc-700 dark:bg-zinc-950 dark:focus:border-zinc-300"
    />
  );
}

function Select({
  value,
  onChange,
  options,
}: {
  value: string;
  onChange: (v: string) => void;
  options: string[];
}) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="rounded-lg border border-zinc-300 bg-white px-3 py-2 text-sm outline-none focus:border-zinc-900 dark:border-zinc-700 dark:bg-zinc-950 dark:focus:border-zinc-300"
    >
      {!options.includes(value) && <option value={value}>{value}</option>}
      {options.map((o) => (
        <option key={o} value={o}>
          {o}
        </option>
      ))}
    </select>
  );
}

function Row({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex justify-between gap-4">
      <dt className="text-zinc-500 dark:text-zinc-400">{k}</dt>
      <dd className="font-mono">{v}</dd>
    </div>
  );
}