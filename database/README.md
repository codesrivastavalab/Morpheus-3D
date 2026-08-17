# Database

The Morpheus-3D database is not included in this repository because of its size.

Download the database from:

https://huggingface.co/datasets/sreeharshk/Morpheus3D-Database

Extract `Morpheus3D_database_sqlite.zip` and place the contents in this directory:

```
database/
├── Morpheus3D_database.sqlite
└── ...
```

Then run Morpheus-3D with:

```bash
--db database/Morpheus3D_database.sqlite
```
