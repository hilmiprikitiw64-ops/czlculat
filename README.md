omnicalc-pro/
├── backend/
│ ├── app/
│ │ ├── __init__.py
│ │ ├── main.py
│ │ ├── config.py
│ │ ├── logger.py
│ │ ├── api/
│ │ │ ├── __init__.py
│ │ │ └── v1/
│ │ │ ├── __init__.py
│ │ │ ├── calculators.py
│ │ │ ├── descriptors.py
│ │ │ └── auth.py
│ │ ├── calculators/
│ │ │ ├── __init__.py
│ │ │ ├── base.py
│ │ │ ├── quadratic.py
│ │ │ ├── derivative.py
│ │ │ ├── integral_numeric.py
│ │ │ ├── matrix_inverse.py
│ │ │ ├── matrix_determinant.py
│ │ │ ├── linear_system.py
│ │ │ ├── fourier_dft.py
│ │ │ ├── physics_kinematics.py
│ │ │ ├── unit_convert.py
│ │ │ └── stats_mean_std.py
│ │ ├── services/
│ │ │ ├── math_engine.py
│ │ │ └── persistence.py
│ │ ├── db/
│ │ │ ├── __init__.py
│ │ │ ├── session.py
│ │ │ └── init_db.py
│ │ ├── models/
│ │ │ └── user.py
│ │ ├── schemas/
│ │ │ ├── calculator.py
│ │ │ └── user.py
│ │ ├── utils/
│ │ │ ├── sandbox.py
│ │ │ ├── security.py
│ │ │ └── rate_limit.py
│ │ └── tests/
│ │ ├── test_quadratic.py
│ │ ├── test_derivative.py
│ │ └── test_matrix_inverse.py
│ ├── requirements.txt
│ ├── Dockerfile
│ └── docker-compose.dev.yml
├── frontend/
│ ├── web/
│ │ ├── package.json
│ │ ├── vite.config.ts
│ │ ├── tsconfig.json
│ │ ├── public/
│ │ │ └── favicon.svg
│ │ ├── src/
│ │ │ ├── main.tsx
│ │ │ ├── App.tsx
│ │ │ ├── index.css
│ │ │ ├── design-tokens.json
│ │ │ ├── components/
│ │ │ │ ├── Landing.tsx
│ │ │ │ ├── Dashboard.tsx
│ │ │ │ ├── CalculatorPage.tsx
│ │ │ │ ├── CalculatorPluginLoader.tsx
│ │ │ │ ├── CalculatorForm.tsx
│ │ │ │ └── ResultCard.tsx
│ │ │ ├── modules/
│ │ │ │ ├── quadratic/
│ │ │ │ │ ├── QuadraticUI.tsx
│ │ │ │ │ └── schema.ts
│ │ │ │ ├── derivative/
│ │ │ │ │ ├── DerivativeUI.tsx
│ │ │ │ │ └── schema.ts
│ │ │ │ └── matrix_inverse/
│ │ │ │ ├── MatrixInverseUI.tsx
│ │ │ │ └── schema.ts
│ │ │ ├── hooks/
│ │ │ └── tests/
│ │ │ ├── CalculatorPage.test.tsx
│ │ │ └── QuadraticUI.test.tsx
│ │ └── tailwind.config.cjs
│ ├── Dockerfile
│ └── docker-compose.dev.yml
├── infra/
│ ├── terraform/
│ │ └── minimal-omnicalc.tf
│ └── kubernetes/
│ └── deployment.yaml
├── .github/
│ └── workflows/
│ └── ci.yml
├── docker-compose.yml
├── Makefile
├── .env.example
├── LICENSE
└── README.md
