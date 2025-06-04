## Traditional RAG on + Restructured Datasheet (ragtest0)
#### Knowledge Base Setup & Query CLI
```
uv run main.py
```

## Raw JSON .txt (ragtest1)
#### Knowledge Base Setup CLI
```
graphrag index --root ./ragtest1
```
#### Query CLI
```
graphrag query --root ./ragtest1 --method global --query "What is the operation system of ROM-2820?"
```

## Restructured JSON .json (ragtest2)
#### Knowledge Graph Setup CLI
```
graphrag index --root ./ragtest2
```
#### Query CLI
```
graphrag query --root ./ragtest2 --method global --query "What is the operation system of ROM-2820?"
```

## Restructured JSON .json + allow_general_knowledge (ragtset2)
#### Knowledge Graph Setup CLI (if not setup yet)
```
graphrag index --root ./ragtest3
```
#### Query CLI
```
graphrag query --root ./ragtest3 --method global --query "What is the operation system of ROM-2820?"
```

## Restructured JSON .json + Restructured Datasheet (ragtset2)

## Restructured JSON .json + Restructured Datasheet + allow_general_knowledge