# Video Captioning

В задаче Video Captioning (создание текстового описания к видео) модели необходимо проанализировать короткий видеофрагмент и сгенерировать наиболее подходящее текстовое описание на английском языке, которое характеризует события и/или действия, происходящие на видео.

![image](https://user-images.githubusercontent.com/91266802/215274541-1cabfd9c-dfc0-462b-b582-a4e24efc556b.png)

Generated Caption: 
a woman and her dog on the beach

## Решение
Пайплайн работы модели включает в себя следующие этапы: вычисление для входного видео эмбеддингов с помощью CLIP, которые далее пропускаются через MLP адаптер для GPT декодера и декодер предсказывает ответ.

## Метрика
Используется метрика BLEU, которая позволяет сравнить эталонный и предсказанный текст. При этом BLEU оценивает не только соответствие отдельных слов, но и n-грамм, содержащихся в тексте.
Метрика BLEU была изначально предложена для оценки качества машинного перевода, однако она может применяться в любых задачах, в которых необходимо оценить близость двух текстов (при этом, допуская вариативность текстов-кандидатов, что важно в задаче описания видео).
