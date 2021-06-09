для корректной работы с переменными, созданными из UI (используется cозданная в GUI переменная model_path=/data/models/{{ds}}/model.pkl)
```
export FERNET_KEY=$(python -c "from cryptography.fernet import Fernet; FERNET_KEY = Fernet.generate_key().decode(); print(FERNET_KEY)")
docker compose up --build
```
Скриншоты:

Все даги

![TreeView](screenshots/0_all_dags.png)

- dag_data_generation

![TreeView](screenshots/1_dag_tree.png)
![TreeView](screenshots/1_dag_graph.png)

- dag_train_model
![TreeView](screenshots/2_dag_tree.png)
![TreeView](screenshots/2_dag_graph.png)

- dag_get_predictions
![TreeView](screenshots/3_dag_tree.png)
![TreeView](screenshots/3_dag_graph.png)


