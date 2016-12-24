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





