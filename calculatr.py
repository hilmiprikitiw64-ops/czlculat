Backend — backend/app/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1 import calculators, descriptors
from app.logger import configure_logging

configure_logging()

app = FastAPI(title="OmniCalc Pro API", version="0.2.0")

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(descriptors.router, prefix="/api/v1/descriptor", tags=["descriptor"])
app.include_router(calculators.router, prefix="/api/v1/calculators", tags=["calculators"])

@app.get("/healthz")
async def health():
    return {"status": "ok"}
  Backend — backend/app/api/v1/calculators.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any
from app.calculators import (
    quadratic, derivative, integral_numeric, matrix_inverse,
    matrix_determinant, linear_system, fourier_dft, physics_kinematics,
    unit_convert, stats_mean_std
)
from app.utils.sandbox import run_with_timeout, TimeoutException

router = APIRouter()

# Generic wrapper model
class CalcRequest(BaseModel):
    payload: dict
    steps: bool = False

@router.post("/quadratic")
async def calc_quadratic(req: CalcRequest):
    try:
        return run_with_timeout(lambda: quadratic.solve_quadratic(**req.payload, steps=req.steps), timeout=2)
    except TimeoutException:
        raise HTTPException(504, "Calculation timed out")
    except Exception as e:
        raise HTTPException(400, str(e))

# derivative
@router.post("/derivative")
async def calc_derivative(req: CalcRequest):
    try:
        return run_with_timeout(lambda: derivative.compute_derivative(**req.payload, steps=req.steps), timeout=3)
    except Exception as e:
        raise HTTPException(400, str(e))

@router.post("/integral_numeric")
async def calc_integral(req: CalcRequest):
    try:
        return run_with_timeout(lambda: integral_numeric.integrate_numeric(**req.payload, steps=req.steps), timeout=4)
    except Exception as e:
        raise HTTPException(400, str(e))

@router.post("/matrix_inverse")
async def calc_matrix_inverse(req: CalcRequest):
    try:
        return matrix_inverse.inverse(req.payload.get("matrix"))
    except Exception as e:
        raise HTTPException(400, str(e))

@router.post("/matrix_determinant")
async def calc_matrix_det(req: CalcRequest):
    try:
        return matrix_determinant.determinant(req.payload.get("matrix"))
    except Exception as e:
        raise HTTPException(400, str(e))

@router.post("/linear_system")
async def calc_linear_system(req: CalcRequest):
    try:
        return linear_system.solve_system(req.payload.get("A"), req.payload.get("b"))
    except Exception as e:
        raise HTTPException(400, str(e))

@router.post("/fourier_dft")
async def calc_dft(req: CalcRequest):
    try:
        return fourier_dft.dft(req.payload.get("signal"))
    except Exception as e:
        raise HTTPException(400, str(e))

@router.post("/physics/kinematics")
async def calc_kinematics(req: CalcRequest):
    try:
        return physics_kinematics.compute(**req.payload)
    except Exception as e:
        raise HTTPException(400, str(e))

@router.post("/unit/convert")
async def convert_unit(req: CalcRequest):
    try:
        return unit_convert.convert(req.payload.get("value"), req.payload.get("from_unit"), req.payload.get("to_unit"))
    except Exception as e:
        raise HTTPException(400, str(e))

@router.post("/stats/mean_std")
async def stats_mean_std_endpoint(req: CalcRequest):
    try:
        return stats_mean_std.mean_std(req.payload.get("values"))
    except Exception as e:
        raise HTTPException(400, str(e))
Backend calculators — contoh 10 modul (ringkas)

backend/app/calculators/quadratic.py

from typing import Dict, Any
import math

def solve_quadratic(a: float, b: float, c: float, steps: bool = False) -> Dict[str, Any]:
    if a == 0:
        raise ValueError("a must be non-zero")
    d = b*b - 4*a*c
    steps_list = [f"Compute discriminant d = b^2 - 4ac = {d}"]
    if d > 0:
        r1 = (-b + math.sqrt(d)) / (2*a)
        r2 = (-b - math.sqrt(d)) / (2*a)
        roots = [r1, r2]
        steps_list.append(f"Two real roots: {r1}, {r2}")
    elif d == 0:
        r = -b/(2*a)
        roots = [r]
        steps_list.append(f"One real root (double): {r}")
    else:
        re = -b/(2*a)
        im = math.sqrt(-d)/(2*a)
        roots = [complex(re, im), complex(re, -im)]
        steps_list.append(f"Two complex roots: {roots}")
    out = {"roots": [str(r) for r in roots], "discriminant": d}
    if steps:
        out["steps"] = steps_list
    return out

backend/app/calculators/derivative.py

from app.services.math_engine import derivative as sympy_derivative

def compute_derivative(expr: str, var: str = "x", steps: bool = False):
    # returns symbolic derivative (string) and optionally steps
    d = sympy_derivative(expr, var)
    return {"expression": expr, "derivative": str(d)}

backend/app/calculators/integral_numeric.py

from app.services.math_engine import integrate_numeric

def integrate_numeric(expr: str, var: str = "x", a: float = 0.0, b: float = 1.0, steps: bool = False):
    val = integrate_numeric(expr, var, a, b)
    return {"expression": expr, "a": a, "b": b, "value": val}

backend/app/calculators/matrix_inverse.py

import numpy as np

def inverse(matrix):
    arr = np.array(matrix, dtype=float)
    if arr.shape[0] != arr.shape[1]:
        raise ValueError("Matrix must be square")
    inv = np.linalg.inv(arr)
    return {"inverse": inv.tolist()}

backend/app/calculators/matrix_determinant.py

import numpy as np

def determinant(matrix):
    arr = np.array(matrix, dtype=float)
    if arr.shape[0] != arr.shape[1]:
        raise ValueError("Matrix must be square")
    det = float(np.linalg.det(arr))
    return {"determinant": det}

backend/app/calculators/linear_system.py

import numpy as np

def solve_system(A, b):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    x = np.linalg.solve(A, b)
    return {"solution": x.tolist()}

backend/app/calculators/fourier_dft.py

import numpy as np

def dft(signal):
    x = np.array(signal, dtype=float)
    N = x.size
    X = np.fft.fft(x)
    return {"dft": X.tolist(), "N": int(N)}

backend/app/calculators/physics_kinematics.py

def compute(u: float = None, a: float = None, t: float = None, s: float = None):
    # basic kinematics: v = u + a t ; s = u t + 0.5 a t^2
    out = {}
    if u is not None and a is not None and t is not None:
        v = u + a * t
        s_calc = u*t + 0.5*a*(t**2)
        out.update({"v": v, "s": s_calc})
    else:
        raise ValueError("Provide u, a, t")
    return out

backend/app/calculators/unit_convert.py

UNIT_MAP = {
    ("m","cm"): lambda x: x*100,
    ("cm","m"): lambda x: x/100,
    ("km","m"): lambda x: x*1000,
    ("m","km"): lambda x: x/1000,
    ("kg","g"): lambda x: x*1000,
    ("g","kg"): lambda x: x/1000
}

def convert(value, from_unit, to_unit):
    key = (from_unit, to_unit)
    if key not in UNIT_MAP:
        raise ValueError("Conversion not supported")
    return {"value": value, "from": from_unit, "to": to_unit, "result": UNIT_MAP[key](value)}

backend/app/calculators/stats_mean_std.py

import numpy as np

def mean_std(values):
    arr = np.array(values, dtype=float)
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr, ddof=1))}
Backend — math engine wrapper backend/app/services/math_engine.py

from sympy.parsing.sympy_parser import parse_expr
from sympy import Symbol, diff, integrate
from typing import Any

def safe_parse(expr: str):
    # Basic parse with sympy; production must whitelist names
    e = parse_expr(expr, evaluate=True)
    return e

def derivative(expr: str, var: str = "x"):
    e = safe_parse(expr)
    s = Symbol(var)
    return diff(e, s)

def integrate_numeric(expr: str, var: str = "x", a: float = 0.0, b: float = 1.0):
    e = safe_parse(expr)
    s = Symbol(var)
    res = integrate(e, (s, a, b))
    return float(res.evalf())
  Backend — sandbox backend/app/utils/sandbox.py

import multiprocessing
import traceback

class TimeoutException(Exception):
    pass

def _target(func, q, *args, **kwargs):
    try:
        q.put(("ok", func(*args, **kwargs)))
    except Exception as e:
        q.put(("err", traceback.format_exc()))

def run_with_timeout(func, args=(), kwargs=None, timeout=3):
    if kwargs is None:
        kwargs = {}
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=_target, args=(func, q, *args), kwargs=kwargs)
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        raise TimeoutException("Execution timed out")
    if q.empty():
        raise Exception("No result returned from worker")
    status, payload = q.get()
    if status == "ok":
        return payload
    else:
        raise Exception(payload)
  Backend — logger backend/app/logger.py

import logging
import sys

def configure_logging():
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    root.addHandler(handler)
  Backend — requirements backend/requirements.txt

fastapi
uvicorn[standard]
pydantic
sympy
numpy
scipy
sqlalchemy
psycopg2-binary
redis
python-jose[cryptography]
passlib[bcrypt]
pytest
httpx
gunicorn
Frontend — entry frontend/web/src/main.tsx

import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "./index.css";

createRoot(document.getElementById("root")!).render(<App />);

frontend/web/src/App.tsx

import React from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Landing from "./components/Landing";
import Dashboard from "./components/Dashboard";
import CalculatorPage from "./components/CalculatorPage";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/calc/:module" element={<CalculatorPage />} />
      </Routes>
    </BrowserRouter>
  );
}

frontend/web/src/components/CalculatorPage.tsx (generic plugin loader + UI)

import React, { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import axios from "axios";
import CalculatorForm from "./CalculatorForm";
import ResultCard from "./ResultCard";

export default function CalculatorPage() {
  const { module } = useParams<{ module: string }>();
  const [descriptor, setDescriptor] = useState<any | null>(null);
  const [result, setResult] = useState<any | null>(null);

  useEffect(() => {
    if (!module) return;
    axios.get(`/api/v1/descriptor/${module}`).then((r) => setDescriptor(r.data)).catch(() => {
      setDescriptor({ title: module, inputs: [] });
    });
  }, [module]);

  function handleSubmit(payload: any) {
    axios.post(`/api/v1/calculators/${module}`, { payload, steps: !!payload.steps }).then((r) => {
      setResult(r.data);
    }).catch((err) => {
      setResult({ error: err.response?.data || String(err) });
    });
  }

  if (!descriptor) return <div>Loading...</div>;

  return (
    <div className="p-6 max-w-4xl mx-auto">
      <h1 className="text-2xl font-semibold mb-4">{descriptor.title}</h1>
      <CalculatorForm descriptor={descriptor} onSubmit={handleSubmit} />
      {result && <ResultCard result={result} />}
    </div>
  );
}

frontend/web/src/components/CalculatorForm.tsx (dynamic form)

import React, { useState } from "react";

export default function CalculatorForm({ descriptor, onSubmit }: any) {
  const initial: any = {};
  (descriptor.inputs || []).forEach((inp: any) => (initial[inp.name] = inp.default ?? ""));

  const [form, setForm] = useState(initial);

  function setVal(k: string, v: any) {
    setForm((s: any) => ({ ...s, [k]: v }));
  }

  return (
    <form onSubmit={(e) => { e.preventDefault(); onSubmit(form); }}>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {(descriptor.inputs || []).map((inp: any) => (
          <div key={inp.name}>
            <label className="block text-sm font-medium">{inp.label}</label>
            <input
              className="mt-1 block w-full border rounded px-3 py-2"
              value={form[inp.name]}
              onChange={(e) => setVal(inp.name, inp.type === "number" ? Number(e.target.value) : e.target.value)}
              placeholder={inp.placeholder}
            />
            {inp.help && <p className="text-xs text-gray-500">{inp.help}</p>}
          </div>
        ))}
      </div>
      <div className="mt-4 flex gap-2">
        <button type="submit" className="px-4 py-2 rounded bg-indigo-600 text-white">Compute</button>
        <button type="button" className="px-4 py-2 rounded border" onClick={() => navigator.clipboard?.writeText(JSON.stringify(form))}>Copy Input</button>
      </div>
    </form>
  );
}

frontend/web/src/components/ResultCard.tsx

import React from "react";

export default function ResultCard({ result }: any) {
  return (
    <div className="mt-6 p-4 border rounded bg-white/5">
      <pre className="whitespace-pre-wrap text-sm">{JSON.stringify(result, null, 2)}</pre>
      <div className="mt-2 flex gap-2">
        <button onClick={() => navigator.clipboard?.writeText(JSON.stringify(result))} className="px-3 py-1 rounded border">Copy Result</button>
      </div>
    </div>
  );
}

Example frontend module descriptor endpoint (backend) — backend/app/api/v1/descriptors.py

from fastapi import APIRouter

router = APIRouter()

DESCRIPTORS = {
    "quadratic": {
        "title": "Quadratic Equation Solver",
        "inputs": [
            {"name": "a", "label": "a (coefficient)", "type": "number", "default": 1, "placeholder": "1"},
            {"name": "b", "label": "b (coefficient)", "type": "number", "default": -3, "placeholder": "-3"},
            {"name": "c", "label": "c (coefficient)", "type": "number", "default": 2, "placeholder": "2"},
            {"name": "steps", "label": "Show steps", "type": "boolean", "default": False}
        ]
    },
    "derivative": {
        "title": "Derivative (symbolic)",
        "inputs": [
            {"name": "expr", "label": "Expression (sympy syntax)", "type": "text", "placeholder": "sin(x) * x**2"},
            {"name": "var", "label": "Variable", "type": "text", "default": "x"}
        ]
    },
    "matrix_inverse": {
        "title": "Matrix Inverse",
        "inputs": [
            {"name": "matrix", "label": "Matrix (JSON array)", "type": "text", "placeholder": "[[1,2],[3,4]]"}
        ]
    }
    # add descriptors for other modules...
}

@router.get("/{module}")
async def get_descriptor(module: str):
    return DESCRIPTORS.get(module, {"title": module, "inputs": []})
    Frontend — package.json (snippet)

{
  "name": "omnicalc-web",
  "version": "0.1.0",
  "private": true,
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview",
    "test": "vitest",
    "lint": "eslint . --ext .ts,.tsx"
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-router-dom": "^6.10.0",
    "axios": "^1.4.0",
    "tailwindcss": "^3.4.0"
  },
  "devDependencies": {
    "typescript": "^5.1.0",
    "vite": "^5.0.0",
    "eslint": "^8.0.0",
    "vitest": "^1.2.0"
  }
}
