#%%
import numpy as np

# |0>, |1>
zero = np.array([[1.],
                 [0.]])
um   = np.array([[0.],
                 [1.]])

# <0|, <1|
brazero = zero.T
braum   = um.T

# Gates I, H, X, Z
I = np.array([[1, 0],
              [0, 1]])

H = np.array([[1/np.sqrt(2),  1/np.sqrt(2)],
              [1/np.sqrt(2), -1/np.sqrt(2)]])

X = np.array([[0, 1],
              [1, 0]])

Z = np.array([[1,  0],
              [0, -1]])

#%%
# Função para produto tensorial sucessivo
def produto_tensorial_sucessivo(matrizes):
    if not matrizes:
        raise ValueError("A lista de matrizes não pode ser vazia.")
    resultado = matrizes[0]
    for matriz in matrizes[1:]:
        resultado = np.kron(resultado, matriz)
    return resultado

#%%
# Projetores |0><0| e |1><1|
zerozero = produto_tensorial_sucessivo([zero, brazero])   # |0><0|
umum     = produto_tensorial_sucessivo([um,   braum  ])   # |1><1|

#%%
# Operador Oráculo: marca w = |01>
# O = I - 2|w><w|
r3    = produto_tensorial_sucessivo([zero, um])
r4    = produto_tensorial_sucessivo([brazero, braum])
inner = np.dot(r3, r4)
I2    = produto_tensorial_sucessivo([I, I])
operadorO = I2 - 2 * inner
print("Oráculo (I - 2|01><01|):")
print(np.round(operadorO, 4))

#%%
# Operador de Reflexão: 2|00><00| - I  
r1    = produto_tensorial_sucessivo([zero, zero])
r2    = produto_tensorial_sucessivo([brazero, brazero])
inner = np.dot(r1, r2)
operadorR = 2 * inner - I2
print("\nReflexão (2|00><00| - I):")
print(np.round(operadorR, 4))

#%%
# CZ via produto tensorial
Cz = np.array([[1, 0, 0,  0],
               [0, 1, 0,  0],
               [0, 0, 1,  0],
               [0, 0, 0, -1]])

# Oráculo via X·CZ·X (marca |01>)
X0     = produto_tensorial_sucessivo([X, I])
oraculo_cz = X0 @ Cz @ X0
print("\nOráculo via X·CZ·X:")
print(np.round(oraculo_cz, 4))

#%%
# Operador de Difusão de Grover
# D = H⊗H · (2|00><00| - I) · H⊗H
H2 = produto_tensorial_sucessivo([H, H])
operadorDifusao = H2 @ operadorR @ H2
print("\nOperador de Difusão:")
print(np.round(operadorDifusao, 4))

#%%
# Estado inicial |00>
estado = produto_tensorial_sucessivo([zero, zero])
print("\nEstado inicial |00>:")
print(estado.T)

#%%
# Passo 1: Superposição — H⊗H |00>
estado = H2 @ estado
print("\nApós H⊗H (superposição uniforme):")
print(np.round(estado.T, 4))

#%%
# Iteração de Grover: Oráculo → Difusão
# Para 2 qubits (N=4), 1 iteração é o ótimo: k = floor(pi/4 * sqrt(4)) = 1

# Oráculo
estado = operadorO @ estado
print("\nApós Oráculo:")
print(np.round(estado.T, 4))

# Difusão
estado = operadorDifusao @ estado
print("\nApós Difusão (1 iteração completa):")
print(np.round(estado.T, 4))

#%%
# Probabilidades do estado final
probabilidades = np.abs(estado.flatten()) ** 2
estados_base   = ['|00>', '|01>', '|10>', '|11>']

print("\nProbabilidades:")
for base, prob in zip(estados_base, probabilidades):
    barra = '█' * int(prob * 40)
    print(f"  {base}: {prob:.4f}  {barra}")

estado_marcado = estados_base[np.argmax(probabilidades)]
print(f"\nEstado com maior probabilidade: {estado_marcado}")
print(f"Probabilidade de sucesso: {max(probabilidades):.2%}")



# %%
