#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>

using Complex = std::complex<double>;
using MatrixXc = std::vector<std::vector<Complex>>;
using VectorXc = std::vector<Complex>;

const double PI = std::acos(-1.0);
const double EULER = 0.5772156649015329;
const Complex SIGMA(0.0, -0.25); // 1 / (4i) = -0.25i

struct PontoBarreira {
    int r, c;
    double x, y;
    double gamma;
    int commits;
};

// Resolvedor de Sistema Linear A * x = b em C++ puro (Eliminação Gaussiana com Pivoteamento Parcial)
VectorXc resolverSistemaLinear(MatrixXc A, VectorXc b) {
    int n = b.size();
    for (int i = 0; i < n; ++i) {
        // Busca o pivô máximo na coluna i
        int max_row = i;
        double max_val = std::abs(A[i][i]);
        for (int k = i + 1; k < n; ++k) {
            if (std::abs(A[k][i]) > max_val) {
                max_val = std::abs(A[k][i]);
                max_row = k;
            }
        }
        std::swap(A[i], A[max_row]);
        std::swap(b[i], b[max_row]);

        // Eliminação abaixo do pivô
        for (int k = i + 1; k < n; ++k) {
            Complex fator = A[k][i] / A[i][i];
            b[k] -= fator * b[i];
            for (int j = i; j < n; ++j) {
                A[k][j] -= fator * A[i][j];
            }
        }
    }

    // Substituição reversa
    VectorXc x(n);
    for (int i = n - 1; i >= 0; --i) {
        Complex soma = b[i];
        for (int j = i + 1; j < n; ++j) {
            soma -= A[i][j] * x[j];
        }
        x[i] = soma / A[i][i];
    }
    return x;
}

// Leitor nativo do JSON gerado pela API do GitHub
std::vector<std::vector<int>> lerCommitsDoArquivo(const std::string& caminho_json) {
    std::ifstream file(caminho_json);
    std::vector<std::vector<int>> malha;
    if (!file.is_open()) return malha;

    std::string conteudo((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    malha.resize(7, std::vector<int>(53, 0));

    size_t pos = 0;
    int dia_contador = 0;
    while ((pos = conteudo.find("\"contributionCount\"", pos)) != std::string::npos) {
        size_t colon1 = conteudo.find(":", pos);
        int count = std::stoi(conteudo.substr(colon1 + 1));

        size_t pos_wd = conteudo.find("\"weekday\"", colon1);
        size_t colon2 = conteudo.find(":", pos_wd);
        int weekday = std::stoi(conteudo.substr(colon2 + 1));

        int col = dia_contador / 7;
        if (col < 53 && weekday >= 0 && weekday < 7) {
            malha[weekday][col] = count;
        }
        if (weekday == 6) {
            dia_contador = ((dia_contador / 7) + 1) * 7;
        } else {
            dia_contador++;
        }
        pos = colon2;
    }
    return malha;
}

// Integral analítica da diagonal principal M_ii (sub-rotina dgreta)
Complex dgreta(double ds, double k) {
    int n = 0;
    double den = 1.0, parc = 0.0;
    Complex int_val(0.0, 0.0);

    while (n < 100) {
        if (n > 0) {
            den *= std::pow(2.0 * n, 2);
            parc += 1.0 / n;
        }
        double y1 = (2.0 / (2.0 * n + 1.0)) * std::pow(ds / 2.0, 2 * n + 1) * std::pow(k, 2 * n) / den;
        double y2 = (2.0 / PI) * y1 * (EULER - parc - 1.0 / (2.0 * n + 1.0) + std::log(k * ds / 4.0));
        
        int_val += (n % 2 == 0 ? 1.0 : -1.0) * Complex(y1, y2);
        if (std::abs(y1) <= 1e-20 && std::abs(y2) <= 1e-20) break;
        n++;
    }
    return SIGMA * int_val;
}

// Garante compatibilidade da função de Bessel entre Windows (MSVC) e Linux (GCC no GitHub Actions)
#ifndef _WIN32
#define _j0 j0
#define _y0 y0
#endif

// Função de Green livre 2D compatível com qualquer versão de C++ no Windows
Complex G0_livre(double x1, double y1, double x2, double y2, double k) {
    double dist = std::hypot(x1 - x2, y1 - y2);
    if (dist < 1e-12) return Complex(0.0, 0.0);
    double kr = k * dist;
    // _j0 e _y0 são as funções nativas de Bessel/Neumann no Windows (<cmath>)
    return SIGMA * Complex(_j0(kr), _y0(kr));
}

// Onda plana incidente: phi(r) = exp(i * k * (x*cos(theta) + y*sin(theta)))
Complex ondaIncidente(double x, double y, double k, double angulo_inc) {
    double fase = k * (x * std::cos(angulo_inc) + y * std::sin(angulo_inc));
    return std::exp(Complex(0.0, fase));
}

// Mapeia a amplitude instantânea Re(Psi * e^{-iwt}) em cores Hexadecimais (sem std::clamp)
std::string mapearCorOnda(double valor_onda, int commits) {
    double val = (valor_onda + 1.4) / 2.8;
    double norm = std::max(0.0, std::min(1.0, val));

    int r, g, b;
    if (commits > 0) {
        r = static_cast<int>(15 + norm * 40);
        g = static_cast<int>(80 + norm * 175);
        b = static_cast<int>(100 + norm * 155);
    } else {
        r = static_cast<int>(22 + norm * 25);
        g = static_cast<int>(27 + norm * 105);
        b = static_cast<int>(34 + norm * 200);
    }

    std::stringstream ss;
    ss << "#" << std::hex << std::setfill('0')
       << std::setw(2) << r << std::setw(2) << g << std::setw(2) << b;
    return ss.str();
}

// Exporta o arquivo SVG animado
void exportarSVGAnimado(const MatrixXc& psi_grid, const std::vector<std::vector<int>>& malha_commits, const std::string& nome_arquivo) {
    int linhas = psi_grid.size();
    int colunas = psi_grid[0].size();
    int num_frames = 24;

    int tam_celula = 11;
    int espacamento = 3;
    int margem = 15;
    int largura_svg = margem * 2 + colunas * (tam_celula + espacamento);
    int altura_svg = margem * 2 + linhas * (tam_celula + espacamento);

    std::ofstream svg(nome_arquivo);
    svg << "<svg width=\"" << largura_svg << "\" height=\"" << altura_svg 
        << "\" viewBox=\"0 0 " << largura_svg << " " << altura_svg 
        << "\" xmlns=\"http://www.w3.org/2000/svg\">\n";
    svg << "  <rect width=\"100%\" height=\"100%\" fill=\"#0d1117\" rx=\"6\"/>\n";

    for (int r = 0; r < linhas; ++r) {
        for (int c = 0; c < colunas; ++c) {
            int x_px = margem + c * (tam_celula + espacamento);
            int y_px = margem + r * (tam_celula + espacamento);

            std::string valores_cores = "";
            for (int f = 0; f <= num_frames; ++f) {
                double omega_t = (2.0 * PI * f) / num_frames;
                Complex fator_temporal = std::exp(Complex(0.0, -omega_t));
                double I_rt = std::real(psi_grid[r][c] * fator_temporal);
                
                valores_cores += mapearCorOnda(I_rt, malha_commits[r][c]);
                if (f < num_frames) valores_cores += ";";
            }

            svg << "  <rect x=\"" << x_px << "\" y=\"" << y_px 
                << "\" width=\"" << tam_celula << "\" height=\"" << tam_celula 
                << "\" rx=\"2\" fill=\"#161b22\"";
            
            if (malha_commits[r][c] > 0) {
                svg << " stroke=\"#58a6ff\" stroke-width=\"0.6\" stroke-opacity=\"0.5\"";
            }
            svg << ">\n";
            svg << "    <animate attributeName=\"fill\" values=\"" << valores_cores 
                << "\" dur=\"2.2s\" repeatCount=\"indefinite\" />\n";
            svg << "  </rect>\n";
        }
    }

    svg << "</svg>\n";
    svg.close();
    std::cout << "SVG animado gerado com sucesso em: " << nome_arquivo << "\n";
}

int main() {
    std::vector<std::vector<int>> malha_commits = lerCommitsDoArquivo("commits.json");

    if (malha_commits.empty()) {
        std::cout << "Arquivo 'commits.json' nao encontrado. Usando grade sintetica de teste (7x52).\n";
        malha_commits.resize(7, std::vector<int>(52, 0));
        for (int r = 0; r < 7; ++r) {
            if (r != 2 && r != 4) malha_commits[r][15] = 8; 
        }
        malha_commits[3][28] = 10;
        malha_commits[2][29] = 4;
        malha_commits[4][29] = 4;
    } else {
        std::cout << "Dados reais carregados de 'commits.json' com sucesso!\n";
    }

    int linhas = malha_commits.size();
    int colunas = malha_commits[0].size();

    double ds = 1.0;
    double k = 1.4;
    double angulo_inc = 0.08;
    double escala_gamma = 1.5;

    std::vector<PontoBarreira> barreira;
    for (int r = 0; r < linhas; ++r) {
        for (int c = 0; c < colunas; ++c) {
            if (malha_commits[r][c] > 0) {
                double x = c * ds + ds / 2.0;
                double y = r * ds + ds / 2.0;
                double gamma = escala_gamma * malha_commits[r][c];
                barreira.push_back({r, c, x, y, gamma, malha_commits[r][c]});
            }
        }
    }

    int N = barreira.size();
    std::cout << "Pontos espalhadores ativos (dias com commits): " << N << "\n";

    Complex diag_val = dgreta(ds, k);
    VectorXc pesos_espalhamento(N, Complex(0.0, 0.0));

    if (N > 0) {
        MatrixXc M(N, VectorXc(N, Complex(0.0, 0.0)));
        VectorXc Phi(N);

        for (int i = 0; i < N; ++i) {
            Phi[i] = ondaIncidente(barreira[i].x, barreira[i].y, k, angulo_inc);
            for (int j = 0; j <= i; ++j) {
                if (i == j) {
                    M[i][i] = diag_val;
                } else {
                    Complex g0 = G0_livre(barreira[i].x, barreira[i].y, barreira[j].x, barreira[j].y, k);
                    M[i][j] = g0 * ds;
                    M[j][i] = M[i][j];
                }
            }
        }

        // Monta A = I - M * diag(gamma)
        MatrixXc A(N, VectorXc(N, Complex(0.0, 0.0)));
        for (int i = 0; i < N; ++i) {
            A[i][i] = Complex(1.0, 0.0);
            for (int j = 0; j < N; ++j) {
                A[i][j] -= M[i][j] * barreira[j].gamma;
            }
        }

        VectorXc psi_fronteira = resolverSistemaLinear(A, Phi);
        for (int j = 0; j < N; ++j) {
            pesos_espalhamento[j] = barreira[j].gamma * psi_fronteira[j];
        }
    }

    MatrixXc psi_grid(linhas, VectorXc(colunas, Complex(0.0, 0.0)));
    for (int r = 0; r < linhas; ++r) {
        for (int c = 0; c < colunas; ++c) {
            double x = c * ds + ds / 2.0;
            double y = r * ds + ds / 2.0;

            Complex psi = ondaIncidente(x, y, k, angulo_inc);
            for (int j = 0; j < N; ++j) {
                if (r == barreira[j].r && c == barreira[j].c) {
                    psi += diag_val * pesos_espalhamento[j];
                } else {
                    psi += G0_livre(x, y, barreira[j].x, barreira[j].y, k) * ds * pesos_espalhamento[j];
                }
            }
            psi_grid[r][c] = psi;
        }
    }

    exportarSVGAnimado(psi_grid, malha_commits, "github-wave.svg");
    return 0;
}