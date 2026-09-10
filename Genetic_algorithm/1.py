import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation




def fitness_function(Osob):
    return -(np.cos(Osob[0]) * np.cos(Osob[1]) * np.exp(Osob[1] / 2))


def generate_population(size, x_range, y_range):
    return [(np.random.uniform(*x_range), np.random.uniform(*y_range)) for _ in range(size)]


def mutation(osobi_2, x_range=(-2, 2), y_range=(-2, 2)):
    result_population = []

    # Скрещивание (среднее арифметическое двух лучших)
    a, b = osobi_2[0]
    c, d = osobi_2[1]
    x_result = np.clip((a + c) / 2, *x_range)
    y_result = np.clip((b + d) / 2, *y_range)
    result_population.append((x_result, y_result))

    # Добавляем 3 мутировавших варианта лучших особей
    for _ in range(3):
        x_mut = np.clip(x_result + np.random.uniform(-0.3, 0.3), *x_range)
        y_mut = np.clip(y_result + np.random.uniform(-0.3, 0.3), *y_range)
        result_population.append((x_mut, y_mut))

    return result_population



x_vals = np.linspace(-2, 2, 200)
y_vals = np.linspace(-2, 2, 200)
X, Y = np.meshgrid(x_vals, y_vals)
Z = -(fitness_function((X, Y)))


generations = 20
population_size = 4

population_history = []  # Список для хранения всех поколений

# Генерация начальной популяции
population = generate_population(population_size, (-2, 2), (-2, 2))

for generation in range(generations):
    population = sorted(population, key=fitness_function)
    population_history.append(population.copy())
    best_population = population[:2]
    population = mutation(best_population)



fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_xlabel("X координата")
ax.set_ylabel("Y координата")


contour = ax.imshow(Z, extent=[-2, 2, -2, 2], origin='lower', cmap='coolwarm', alpha=0.7)


plt.colorbar(contour, label="Значение функции")

title = ax.set_title("Эволюция популяции: Поколение 0")
scatter = ax.scatter([], [], color='black')


def update(frame):
    if frame >= generations - 1:
        ani.event_source.stop()
    pop = population_history[frame]
    x_vals, y_vals = zip(*pop)
    scatter.set_offsets(np.c_[x_vals, y_vals])
    title.set_text(f"Эволюция популяции: Поколение {frame}")
    return scatter, title


ani = animation.FuncAnimation(fig, update, frames=generations, interval=100, blit=False)

ani.save("evolution_2.gif", writer="pillow", fps=5)

plt.show()

