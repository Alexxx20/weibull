import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.stats import weibull_min, expon
import numpy as np
from scipy.optimize import minimize
from statsmodels.distributions.empirical_distribution import ECDF

# Leitura dos dados
df = pd.read_csv('caucaia.txt', delimiter=';')

# Função para calcular as datas e intervalos entre chuvas fortes
def calcular_datas_e_intervalos(df):
    datas_chuva_forte = []
    
    for index, row in df.iterrows():
        ano = int(row['Anos'])
        mes = int(row['Meses'])
        
        for dia in range(1, 32):
            coluna_dia = f'Dia{dia}'
            if coluna_dia in row and row[coluna_dia] > 100:  # Chuva forte > 100mm
                data = datetime(day=dia, month=mes, year=ano)
                datas_chuva_forte.append(data)
    
    intervalos = []
    for i in range(1, len(datas_chuva_forte)):
        intervalo = (datas_chuva_forte[i] - datas_chuva_forte[i-1]).days
        intervalos.append(intervalo)
    
    return datas_chuva_forte, intervalos

# Executar a função para obter as datas e intervalos
datas_chuva_forte, intervalos = calcular_datas_e_intervalos(df)

# Ajuste do WGRP (Weibull-based Generalized Renewal Process)
def virtual_age_type_I(v_prev, x, q):
    """Kijima Type I virtual age model"""
    return v_prev + q * x

def virtual_age_type_II(v_prev, x, q):
    """Kijima Type II virtual age model"""
    return q * (v_prev + x)

def wgrp_log_likelihood(params, intervalos, virtual_age_func):
    """Função de log-verossimilhança para WGRP"""
    alpha, beta, q = params
    n = len(intervalos)
    ll = n * (np.log(beta) - beta * np.log(alpha))
    
    v = 0  # idade virtual inicial
    for i in range(n):
        x = intervalos[i]
        term1 = (beta - 1) * np.log(x + v)
        term2 = ((x + v)**beta - v**beta) / (alpha**beta)
        ll += term1 - term2
        
        # Atualiza a idade virtual para o próximo intervalo
        v = virtual_age_func(v, x, q)
    
    return -ll  # Retornamos o negativo para minimização

# Ajuste inicial da Weibull padrão para obter parâmetros iniciais
params_weibull = weibull_min.fit(intervalos, floc=0)
initial_alpha = params_weibull[2]  # parâmetro de escala
initial_beta = params_weibull[0]   # parâmetro de forma
initial_q = 0  # valor inicial para q (entre 0 e 1)

# Ajustar o modelo WGRP Type I
bounds = [(0.1, None), (0.1, None), (0, 1)]  # alpha > 0, beta > 0, q entre 0 e 1
result_typeI = minimize(wgrp_log_likelihood, 
                       [initial_alpha, initial_beta, initial_q], 
                       args=(intervalos, virtual_age_type_I),
                       bounds=bounds,
                       method='L-BFGS-B')

alpha_I, beta_I, q_I = result_typeI.x

# Ajustar o modelo WGRP Type II
result_typeII = minimize(wgrp_log_likelihood, 
                        [initial_alpha, initial_beta, initial_q], 
                        args=(intervalos, virtual_age_type_II),
                        bounds=bounds,
                        method='L-BFGS-B')

alpha_II, beta_II, q_II = result_typeII.x

# Imprimir resultados
print("=== WGRP Type I ===")
print(f"Alpha: {alpha_I:.2f}")
print(f"Beta: {beta_I:.2f}")
print(f"q: {q_I:.2f}")
print(f"Log-Likelihood: {-result_typeI.fun:.2f}")

print("\n=== WGRP Type II ===")
print(f"Alpha: {alpha_II:.2f}")
print(f"Beta: {beta_II:.2f}")
print(f"q: {q_II:.2f}")
print(f"Log-Likelihood: {-result_typeII.fun:.2f}")

# Teste de qualidade de ajuste usando a transformação de potência WGRP
def wgrp_power_transform(intervalos, alpha, beta, q, virtual_age_func):
    """Aplica a transformação de potência WGRP para obter variáveis exponenciais"""
    W = []
    v = 0
    for x in intervalos:
        w = (x + v)**beta - v**beta
        W.append(w)
        v = virtual_age_func(v, x, q)
    return W

# Aplicar a transformação para ambos os modelos
W_I = wgrp_power_transform(intervalos, alpha_I, beta_I, q_I, virtual_age_type_I)
W_II = wgrp_power_transform(intervalos, alpha_II, beta_II, q_II, virtual_age_type_II)

# Teste de Kolmogorov-Smirnov para exponencialidade
def ks_test_exp(data, theta=1):
    """Teste KS para distribuição exponencial com parâmetro theta"""
    ecdf = ECDF(data)
    x = np.linspace(min(data), max(data), 1000)
    d = np.max(np.abs(ecdf(x) - expon.cdf(x, scale=theta)))
    # Valor-p aproximado (para grandes amostras)
    n = len(data)
    p_value = np.exp(-2 * n * d**2)
    return d, p_value

# Calcular theta estimado (média dos W transformados)
theta_I = np.mean(W_I)
theta_II = np.mean(W_II)

# Executar testes KS
d_I, p_I = ks_test_exp(W_I, theta_I)
d_II, p_II = ks_test_exp(W_II, theta_II)

print("\n=== Teste de Qualidade de Ajuste ===")
print(f"WGRP Type I - KS Statistic: {d_I:.3f}, p-value: {p_I:.3f}")
print(f"WGRP Type II - KS Statistic: {d_II:.3f}, p-value: {p_II:.3f}")

# Selecionar o melhor modelo com base na log-verossimilhança
if result_typeI.fun < result_typeII.fun:
    best_model = "Type I"
    alpha, beta, q = alpha_I, beta_I, q_I
    virtual_age_func = virtual_age_type_I
else:
    best_model = "Type II"
    alpha, beta, q = alpha_II, beta_II, q_II
    virtual_age_func = virtual_age_type_II

print(f"\nMelhor modelo: WGRP {best_model}")

# Interpretação dos parâmetros
print("\n=== Interpretação dos Parâmetros ===")
if beta < 1:
    print("Sistema em melhoria (Beta < 1)")
elif beta > 1:
    print("Sistema em deterioração (Beta > 1)")
else:
    print("Sistema estável (Beta ≈ 1)")

print(f"Parâmetro de rejuvenescimento q = {q:.2f}:")
if q == 0:
    print("sistema volta ao estado 'como novo'")
elif q == 1:
    print("sistema permanece 'tão ruim quanto velho'")
else:
    print(f"estado intermediário entre 'como novo' e 'tão ruim quanto velho'")

# Previsão da próxima chuva forte usando simulação Monte Carlo
def simulate_next_event(datas_chuva_forte, intervalos, alpha, beta, q, virtual_age_func, n_sim=10000):
    """Simula a próxima chuva forte usando WGRP"""
    # Calcular idade virtual atual
    v = 0
    for x in intervalos:
        v = virtual_age_func(v, x, q)
    
    # Última data de chuva forte
    last_date = datas_chuva_forte[-1]
    
    # Simular tempos até o próximo evento
    simulated_times = []
    for _ in range(n_sim):
        u = np.random.uniform()
        x = alpha * (v**beta - np.log(u))**(1/beta) - v
        simulated_times.append(x)
    
    # Calcular estatísticas
    median_time = np.median(simulated_times)
    mean_time = np.mean(simulated_times)
    ci_low = np.percentile(simulated_times, 2.5)
    ci_high = np.percentile(simulated_times, 97.5)
    
    # Converter para datas
    median_date = last_date + timedelta(days=int(median_time))
    mean_date = last_date + timedelta(days=int(mean_time))
    ci_low_date = last_date + timedelta(days=int(ci_low))
    ci_high_date = last_date + timedelta(days=int(ci_high))
    
    return {
        'median_time': median_time,
        'mean_time': mean_time,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'median_date': median_date,
        'mean_date': mean_date,
        'ci_low_date': ci_low_date,
        'ci_high_date': ci_high_date
    }

# Executar simulação
sim_results = simulate_next_event(datas_chuva_forte, intervalos, alpha, beta, q, virtual_age_func)

# Resultados da previsão
print("\n=== Previsão da Próxima Chuva Forte ===")
print(f"Data mediana prevista: {sim_results['median_date'].strftime('%d/%m/%Y')}")
print(f"Data média prevista: {sim_results['mean_date'].strftime('%d/%m/%Y')}")
print(f"Intervalo de confiança 95%: {sim_results['ci_low_date'].strftime('%d/%m/%Y')} a {sim_results['ci_high_date'].strftime('%d/%m/%Y')}")

# Calcular dias até a próxima chuva forte a partir de hoje
data_hoje = datetime.now()
dias_ate_mediana = (sim_results['median_date'] - data_hoje).days

print(f"\nDias até a próxima chuva forte (mediana): {dias_ate_mediana} dias")

# Plotar resultados
plt.figure(figsize=(14, 6))

# Gráfico 1: Distribuição dos intervalos observados vs ajuste WGRP
plt.subplot(1, 2, 1)
x = np.linspace(0, max(intervalos)*1.2, 1000)

# Calcular PDF empírica
hist, bins = np.histogram(intervalos, bins=20, density=True)
bin_centers = (bins[:-1] + bins[1:])/2
plt.bar(bin_centers, hist, width=bins[1]-bins[0], alpha=0.6, label='Dados observados')

# Calcular PDF do WGRP (aproximação)
# Para simplificar, mostramos a PDF da Weibull com os parâmetros ajustados
pdf = weibull_min.pdf(x, beta, scale=alpha)
plt.plot(x, pdf, 'r-', label=f'WGRP PDF (Beta={beta:.2f}, Eta={alpha:.2f}, q={q:.2f})')

plt.title('Distribuição dos Intervalos entre Chuvas Fortes')
plt.xlabel('Dias entre eventos')
plt.ylabel('Densidade de probabilidade')
plt.legend()
plt.grid(True)

# Gráfico 2: Função de risco acumulada
plt.subplot(1, 2, 2)
v = 0
hazard = []
cum_hazard = 0
hazard_points = []
for x_val in intervalos:
    h = (beta/alpha**beta) * (x_val + v)**(beta-1)
    hazard.append(h)
    cum_hazard += h
    hazard_points.append(cum_hazard)
    v = virtual_age_func(v, x_val, q)

plt.plot(range(1, len(intervalos)+1), hazard_points, 'b-', label='Função de risco acumulada')
plt.xlabel('Número do evento')
plt.ylabel('Risco acumulado')
plt.title('Função de Risco Acumulada do WGRP')
plt.legend()
plt.grid(True)

# plt.tight_layout()
# plt.show()

# Novo gráfico: CDF da Weibull com marcações limpas
plt.figure(figsize=(8, 5))
x = np.linspace(0, max(intervalos) * 1.2, 1000)
cdf = weibull_min.cdf(x, beta, scale=alpha)

plt.plot(x, cdf, 'b-', label='CDF Weibull')
plt.xlabel('Dias entre eventos')
plt.ylabel('Probabilidade acumulada')
plt.title('CDF da Distribuição de Intervalos entre Chuvas Fortes')
plt.grid(True)

# Mediana e percentil de 95%
mediana_x = weibull_min.ppf(0.5, beta, scale=alpha)
p95_x = weibull_min.ppf(0.95, beta, scale=alpha)

# Marcar com pontos e anotar
plt.scatter([mediana_x], [0.5], color='orange', zorder=5)
plt.annotate(f'Mediana ≈ {mediana_x:.1f} dias',
             xy=(mediana_x, 0.5),
             xytext=(mediana_x + 100, 0.55),
             arrowprops=dict(facecolor='orange', arrowstyle='->'),
             color='orange')

plt.scatter([p95_x], [0.95], color='red', zorder=5)
plt.annotate(f'95º percentil ≈ {p95_x:.1f} dias',
             xy=(p95_x, 0.95),
             xytext=(p95_x - 500, 0.9),
             arrowprops=dict(facecolor='red', arrowstyle='->'),
             color='red')

plt.legend()
plt.tight_layout()
plt.show()