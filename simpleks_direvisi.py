import numpy as np

def get_input():
    try:
        # Input koefisien fungsi tujuan
        c = list(map(float, input("Masukkan koefisien fungsi tujuan (dipisahkan dengan spasi): ").split()))
        
        # Input jumlah batasan
        m = int(input("Masukkan jumlah batasan: "))
        
        # Input koefisien dan nilai kanan batasan
        A = []
        b = []
        print("\nMasukkan koefisien dan nilai kanan untuk setiap batasan:")
        for i in range(m):
            constraint = list(map(float, input(f"Batasan ke-{i+1} (koefisien dan nilai kanan dipisahkan dengan spasi): ").split()))
            A.append(constraint[:-1])
            b.append(constraint[-1])
        
        return np.array(c), np.array(A), np.array(b)
    
    except Exception as e:
        print(f"\nError saat input: {str(e)}")
        return None, None, None

def revised_simplex(c, A, b):
    m, n = A.shape
    I = np.eye(m)
    A = np.hstack((A, I))
    c = np.concatenate((c, np.zeros(m)))
    basis = list(range(n, n + m))
    
    iteration = 0
    while iteration < 100:
        B = A[:, basis]
        try:
            B_inv = np.linalg.inv(B)
        except np.linalg.LinAlgError:
            print("Error: Matriks basis tidak dapat diinvers")
            return None, None
        
        x_B = B_inv @ b
        c_B = c[basis]
        y = c_B @ B_inv
        reduced_costs = c - y @ A
        
        if all(reduced_costs <= 1e-10):
            x = np.zeros(len(c))
            x[basis] = x_B
            return np.round(x[:n], 0), np.round(c[:n] @ x[:n], 0)
        
        entering = np.argmax(reduced_costs)
        d = B_inv @ A[:, entering]
        
        if all(d <= 0):
            print("Masalah tidak terbatas")
            return None, None
        
        ratios = np.where(d > 0, x_B / d, np.inf)
        if all(np.isinf(ratios)):
            print("Masalah tidak terbatas")
            return None, None
        
        leaving_idx = np.argmin(ratios)
        leaving = basis[leaving_idx]
        basis[leaving_idx] = entering
        
        iteration += 1
    
    print("Peringatan: Mencapai batas maksimum iterasi")
    return None, None

def main():
    print("\n=== PROGRAM SIMPLEKS DIREVISI ===")
    print("\nProgram ini akan menyelesaikan masalah optimisasi linear menggunakan metode Simpleks Direvisi.")
    print("")
    
    # Meminta input dari pengguna
    c, A, b = get_input()
    if c is None:
        return
    
    print("\nMemulai perhitungan...")
    x, z = revised_simplex(c, A, b)
    
    if x is not None:
        print("\n=== Solusi Optimal ===")
        print(f"X1 = {x[0]:.0f}")
        print(f"X2 = {x[1]:.0f}")
        print(f"Z maksimum = {z:.0f}")
    else:
        print("\nTidak ada solusi optimal ditemukan.")
    
if __name__ == "__main__":
    main()