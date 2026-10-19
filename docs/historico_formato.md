# Formato do HISTORY.md

Uma entrada por task (ou subdivisão), em ordem cronológica. Template:

```markdown
## [ID] AAAA-MM-DDTHH:MMZ
- Agente: <identificador do modelo/executor> | Humano: <none ou nome + intervenção>
- Ação: <o que foi feito, 1–2 linhas>
- Resultado: <commits, arquivos, métricas ou "em andamento">
- Próxima: <ID da próxima task>
```

Regras:

- `ID` = prefixo da task no PLAN.md (ex.: `DIR-01`, `SCR-06`).
- Timestamp em UTC, formato ISO (`date -u +%Y-%m-%dT%H:%MZ`).
- Se houve intervenção humana, descrever brevemente (ex.: `Humano: lennin — forneceu path do dataset`).
- Subdivisões de task grande: `SCR-06a`, `SCR-06b`.
- Toda entrada deve apontar a `Próxima` para permitir retomada sem ler o repo inteiro.
