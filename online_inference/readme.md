
Использование модели

Из репозитория:
```
git clone https://github.com/made-ml-in-prod-2021/danidarya.git
cd danidarya
git checkout homework2
cd online_inference
docker build -t danidarya/online_inference:v2 .
docker run -p 8000:8000 danidarya/online_inference:v2
```

Из docker hub:
```
docker pull danidarya/online_inference:v2
docker run -p 8000:8000 danidarya/online_inference:v2
```
Скрипт, делающий запросы:
```
python make_request.py
```

Тесты:
```
pytest tests
```

Оптимизация размера docker образа:
использование python 3.6-slim вместо python 3.6 дало уменьшение размера образа в 2 раза с 495 мб до 214 мб.
[ссылка](https://hub.docker.com/r/danidarya/online_inference/tags?page=1&ordering=last_updated)

Самооценка:
1) Оберните inference вашей модели в rest сервис, должен быть endpoint /predict (3 балла) +

2) Напишите тест для /predict  (3 балла) +

3) Напишите скрипт, который будет делать запросы к вашему сервису (2 балла) +

4) Сделайте валидацию входных данных (3 доп балла) -
 -- возращайте 400, в случае, если валидация не пройдена

5) Напишите dockerfile, соберите на его основе образ и запустите локально контейнер(docker build, docker run),
внутри контейнера должен запускать сервис, написанный в предущем пункте, закоммитьте его,
напишите в readme корректную команду сборки (4 балл)  +

6) Оптимизируйте размер docker image (3 доп балла) +

7) опубликуйте образ в https://hub.docker.com/, используя docker push (вам потребуется зарегистрироваться) (2 балла) +

8) напишите в readme корректные команды docker pull/run, которые должны привести к тому, что локально поднимется
на inference ваша модель (1 балл) +
Убедитесь, что вы можете протыкать его скриптом из пункта 3

9) проведите самооценку -- 1 доп балл +

Итого 19 баллов

