#Ne pas oublier de set le bon working directory !
setwd('/home/samy/workspaces/projets_mlds/AppSup_EtudeComparative/')

########## Données flame ##########
#Chargement des données
flame = read.table('data/flame.txt')

#On scinde la table en deux parties par les colonnes variables et classes
flame_var = flame[,1:2] #Les deux variables
flame_cla = flame[,3] #Les classes (2 classes)

#On plot les variables et on colries en fonction des classes
plot(flame_var,col=flame_cla)
#La on voit bien les deux classes séparées par une courbe parabolique

#Il va falloir bien séparer les ensembles de test et de train
#On va prendre 20 individus comme échantillon de test 
#À moitié de la classe 1 et moitié de la classe 2
test_flame_var = flame_var[144:163,]
train_flame_var = rbind(flame_var[1:143,],flame_var[164:240,])

test_flame_cla = flame_cla[144:163]
train_flame_cla = c(flame_cla[1:143],flame_cla[164:240])

#La précision semble être une bonne mesure pour ésitmer les méthodes sur ce jeu de données
#Le nombre d'individu par classe est assez similaire

#LDA
library(MASS)
flame_z_lda <- lda(train_flame_var, train_flame_cla) #Construction de discrimination linéaire
flame_zp_lda <- predict(flame_z_lda,test_flame_var) #Prediction selon le modèle linéaire
#Table de confusion
table(test_flame_cla,flame_zp_lda$class) #Nous montre une précision très basse de 25%
#Précision que l'on calcule
accurcy_lda_flame = 1 - ( sum(flame_zp_lda$class != test_flame_cla) / length(test_flame_cla) )

#KNN
library(class)
# Ici le bon K n'est pas pré déterminé
# On va boucler sur différents K de 1 à 10 et on sauvegarde la meilleure performance
best_k = 0
best_KNN_acc_flame = 0
for (i in 1:10) {
  flame_knn = knn(train_flame_var, test_flame_var, cl = train_flame_cla, k = i)
  tmp_acc = 1 - ( sum(flame_knn != test_flame_cla) / length(test_flame_cla) )
  if(tmp_acc > best_KNN_acc_flame){
    best_KNN_acc_flame = tmp_acc
    best_k = i
  }
}
#Le meilleur K obtenu est 4 et avec 4NN on obtien une précision de 0.9 déjà meilleur que la LDA

#Classificateur baysien naif
library(e1071)
#Création du modèle
MBN_flame = naiveBayes(as.factor(train_flame_cla) ~ ., data = train_flame_var)
flame_zp_MBN = predict(MBN_flame, test_flame_var) #Prédiction à l'aide du modèle pour le tester
accurcy_MBN_flame = 1 - ( sum(flame_zp_MBN != test_flame_cla) / length(test_flame_cla) )
#Et la on obtien un score de 55% mieux que la LDA mais bon pire que KNN

########## données spiral ##########
#Lecture des données spiral
spiral = read.table('data/spiral.txt')

#Séparation de la table en variables et classes
spiral_var = spiral[,1:2]
spiral_cla = spiral[,3] #Il y a trois classes

#Dessin d'un nuage de points pour visualiser la structure de données
plot(spiral_var, col = spiral_cla)
#La forme bien que jolie des données n'est pas un cadeau pour les méthodes linéaires

#Séparation en données de test et données de train
#Pour prendre des éléments random on prend la matrice de base et on permute les lignes aléatoirement
random_spiral = spiral[sample(nrow(spiral)),]
#Puis on sépare ces données mélangés en classes et variables
spiral_var = random_spiral[,1:2]
spiral_cla = random_spiral[,3]

#Puis finalement on séparre la partie test de la partie train
train_spiral_var = spiral_var[1:280,] #Les premier 280 individus pour le train
train_spiral_cla = spiral_cla[1:280]
test_spiral_var  = spiral_var[280:312,] #les 32 autres indivdus pour le test
test_spiral_cla  = spiral_cla[280:312]

#LDA
#Testons la LDA même si la séparation linéaire sur des données comme celles-ci ne vaut rien
spiral_z_lda <- lda(train_spiral_var, train_spiral_cla) #Construction de discrimination linéaire
spiral_zp_lda <- predict(spiral_z_lda,test_spiral_var) #Prediction selon le modèle linéaire
#Table de confusion
table(test_spiral_cla,spiral_zp_lda$class) #Nous montre une précision basse
#Précision que l'on calcule
accurcy_lda_spiral = 1 - ( sum(spiral_zp_lda$class != test_spiral_cla) / length(test_spiral_cla) )
#On obtien donc 1/3 de précision, ça me surprend qu'il puisse trouver mieux que sur les donnés flame

#KNN
#La encore pas de K pré défini donc boucle de 1 à 10
best_k_spiral = 0
best_KNN_acc_spiral = 0
for (i in 1:10) {
  spiral_knn = knn(train_spiral_var, test_spiral_var, cl = train_spiral_cla, k = i)
  tmp_acc = 1 - ( sum(spiral_knn != test_spiral_cla) / length(test_spiral_cla) )
  if(tmp_acc > best_KNN_acc_spiral){
    best_KNN_acc_spiral = tmp_acc
    best_k_spiral = i
  }
}
#KNN avec un seul voisin (k=1) s'en sort parfaitement et donne un précision de 1 !
#On voit bien sur le scatter plot des données que le plus proche élément
#a de trés forte chances d'appartenir à la même classe

