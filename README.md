# TP-GPGPU

Ce dépôt contient l'ensemble des fichiers relatifs à l'examen final du cours de GPGPU. 

Pour compiler le code et le tester, utilisez la commande suivante :

```bash
nvcc -o tp main.cu matrix.cu ann.cu mnist.cu cudaMatrix.cu 
./tp.exe
``` 

(Il ne faut pas oublier **`cudaMatrix.cu`**, qui est un nouveau fichier !)

Pour lire le compte-rendu du TP, cliquer [ici](rapport.md)
