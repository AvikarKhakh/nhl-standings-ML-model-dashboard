import { useEffect, useMemo, useState } from "react";
import { API_BASE } from "./config";
import "./App.css";

function StatCard({ label, value }) {
  return (
    <div className="card">
      <div className="muted">{label}</div>
      <div className="big">{value}</div>
    </div>
  );
}

function Table({ rows }) {
  return (
    <div className="card">
      <h3 className="h3">Predicted Standings</h3>
      <div className="tableWrap">
        <table>
          <thead>
            <tr>
              <th style={{ width: 60 }}>Rank</th>
              <th>Team</th>
              <th style={{ width: 120 }}>Points</th>
              <th style={{ width: 140 }}>Conference</th>
              <th style={{ width: 140 }}>Division</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={`${r.team}-${r.rank}`}>
                <td>{r.rank}</td>
                <td style={{ fontWeight: 700 }}>{r.team}</td>
                <td>{r.points}</td>
                <td>{r.conference}</td>
                <td>{r.division}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function fmtTeam(t) {
  if (!t) return "—";
  return `${t.team} (${t.points})`;
}

// Playoff format helper (unchanged)
function buildPlayoffBracket(standings) {
  const sortByPoints = (a, b) => Number(b.points) - Number(a.points);

  const byConf = {
    ECF: standings.filter((t) => t.conference === "ECF").slice().sort(sortByPoints),
    WCF: standings.filter((t) => t.conference === "WCF").slice().sort(sortByPoints),
  };

  const confDivisions = {
    ECF: ["Atlantic", "Metro"],
    WCF: ["Central", "Pacific"],
  };

  const top3ByDivision = (teams, div) =>
    teams.filter((t) => t.division === div).slice().sort(sortByPoints).slice(0, 3);

  const buildForConf = (conf) => {
    const teams = byConf[conf];
    const [divA, divB] = confDivisions[conf];

    const a3 = top3ByDivision(teams, divA);
    const b3 = top3ByDivision(teams, divB);

    const locked = new Set([...a3, ...b3].map((t) => t.team));
    const wildcards = teams.filter((t) => !locked.has(t.team)).slice(0, 2);

    const divA1 = a3[0] ?? null;
    const divA2 = a3[1] ?? null;
    const divA3 = a3[2] ?? null;

    const divB1 = b3[0] ?? null;
    const divB2 = b3[1] ?? null;
    const divB3 = b3[2] ?? null;

    const wc1 = wildcards[0] ?? null;
    const wc2 = wildcards[1] ?? null;

    // Best division winner plays WC2
    const winners = [divA1, divB1].filter(Boolean).slice().sort(sortByPoints);
    const bestWinner = winners[0] ?? null;
    const bestWinnerDiv = bestWinner?.division;

    const m1 =
      bestWinnerDiv === divA
        ? { title: `${divA} 1 vs WC2`, a: divA1, b: wc2 }
        : { title: `${divB} 1 vs WC2`, a: divB1, b: wc2 };

    const m2 =
      bestWinnerDiv === divA
        ? { title: `${divB} 1 vs WC1`, a: divB1, b: wc1 }
        : { title: `${divA} 1 vs WC1`, a: divA1, b: wc1 };

    const m3 = { title: `${divA} 2 vs ${divA} 3`, a: divA2, b: divA3 };
    const m4 = { title: `${divB} 2 vs ${divB} 3`, a: divB2, b: divB3 };

    return {
      conference: conf,
      matchups: [m1, m2, m3, m4],
    };
  };

  return { ECF: buildForConf("ECF"), WCF: buildForConf("WCF") };
}

function PlayoffBracket({ standings }) {
  if (!standings || standings.length === 0) return null;
  const brackets = buildPlayoffBracket(standings);

  return (
    <div className="card">
      <h3 className="h3">Playoff Bracket Predictor</h3>

      <div className="bracketGrid">
        {["ECF", "WCF"].map((conf) => (
          <div key={conf} className={`bracketCol ${conf === "ECF" ? "ecf" : "wcf"}`}>
            <div className="bracketTitle">{conf}</div>

            <div className="matchupList">
              {brackets[conf].matchups.map((m) => (
                <div key={m.title} className="matchup">
                  <div className="matchupTitle">{m.title}</div>
                  <div className="matchupTeams">
                    <div>{fmtTeam(m.a)}</div>
                    <div className="vs">vs</div>
                    <div>{fmtTeam(m.b)}</div>
                  </div>
                </div>
              ))}
            </div>

            <div className="muted" style={{ marginTop: 10 }}>
              Top 3 per division + 2 wild cards (by predicted points).
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function RankedListCard({ title, rows, emptyText }) {
  return (
    <div className="card">
      <h3 className="h3">{title}</h3>
      <ol className="list">
        {rows.length ? (
          rows.map((r) => (
            <li key={`${title}-${r.team}-${r.rank}`} className="listItem">
              <span>
                <b>#{r.rank}</b> {r.team}
              </span>
              <span className="pill">{r.points} pts</span>
            </li>
          ))
        ) : (
          <div className="muted">{emptyText}</div>
        )}
      </ol>
    </div>
  );
}

export default function App() {
  const [status, setStatus] = useState("idle");
  const [health, setHealth] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [standings, setStandings] = useState([]);

  const canRun = status !== "running";

  // Blank on load — only health check
  useEffect(() => {
    fetch(`${API_BASE}/api/health`)
      .then((r) => r.json())
      .then((h) => setHealth(h))
      .catch(() => setHealth({ ok: false }));
  }, []);

  const runModel = async () => {
    setStatus("running");
    try {
      const resp = await fetch(`${API_BASE}/api/run`, { method: "POST" }).then((r) => r.json());
      setStandings(resp?.standings || []);
      setMetrics(resp?.metrics || null);
      setStatus("done");
    } catch {
      setStatus("error");
    }
  };

  const clearDashboard = () => {
    setStatus("idle");
    setStandings([]);
    setMetrics(null);
  };

  // ✅ Tier buckets (based on sorted standings)
  const { top5, playoffRest, hunt, rebuild } = useMemo(() => {
    const s = standings.slice(); // already ranked by backend
    const top5 = s.slice(0, 5);
    const playoffTeams = s.slice(0, 16);
    const playoffRest = playoffTeams.slice(5); // exclude top 5 (rest of playoff field)
    const hunt = s.slice(16, 24);
    const rebuild = s.slice(24, 32);
    return { top5, playoffRest, hunt, rebuild };
  }, [standings]);

  return (
    <div className="page">
      <header className="header">
        <div>
          <div className="title">NHL Standings ML Dashboard</div>
          <div className="muted">
            Backend:{" "}
            <span className={health?.ok ? "ok" : "bad"}>
              {health?.ok ? "Connected" : "Offline"}
            </span>
            <span className="dot">•</span> API: {API_BASE}
          </div>
        </div>

        <div className="actions">
          <button className="btn danger" onClick={clearDashboard}>
            Clear
          </button>
          <button className="btn primary" onClick={runModel} disabled={!canRun}>
            {status === "running" ? "Running…" : "Run Model Demo"}
          </button>
        </div>
      </header>

      <section className="grid">
        <StatCard label="RMSE" value={metrics?.rmse ?? "—"} />
        <StatCard label="R²" value={metrics?.r2 ?? "—"} />
        <StatCard label="Last Run" value={metrics?.last_run ?? "—"} />
      </section>

      {/* Left column stack + right table */}
      <section className="grid2">
        <div className="leftStack">
          <RankedListCard
            title="Top 5"
            rows={top5}
            emptyText='No results yet. Click “Run Model Demo”.'
          />

          <RankedListCard
            title="Playoff Teams"
            rows={playoffRest}
            emptyText="Run the model to generate playoff teams."
          />

          <RankedListCard
            title="Teams In The Hunt"
            rows={hunt}
            emptyText="Run the model to generate bubble teams."
          />

          <RankedListCard
            title="Rebuilding Teams"
            rows={rebuild}
            emptyText="Run the model to generate rebuilding teams."
          />
        </div>

        <Table rows={standings} />
      </section>

      <section style={{ marginTop: 14 }}>
        <PlayoffBracket standings={standings} />
      </section>
    </div>
  );
}
