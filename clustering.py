import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from datetime import datetime

def basic_data_overview(df):
    """Aperçu rapide des données"""
    print("📊 APERÇU DES DONNÉES")
    print("="*50)
    print(f"📋 Dimensions: {df.shape[0]} lignes, {df.shape[1]} colonnes")
    print(f"📁 Colonnes disponibles: {list(df.columns)}")
    print("\n🔍 PREMIÈRES LIGNES:")
    print(df.head())
    print("\n📈 STATISTIQUES DESCRIPTIVES:")
    print(df[['Price', 'Purchases', 'NumericRating']].describe())
    print("\n" + "="*50 + "\n")
    return df

def prepare_data_for_clustering(df):
    """Préparation minimale des données pour le clustering"""
    print("🔧 PRÉPARATION DES DONNÉES")
    print("="*50)
    
    df_work = df.copy()
    
    # Renommer les colonnes pour simplifier
    df_work = df_work.rename(columns={
        'App': 'app_name',
        'Price': 'price', 
        'Purchases': 'purchases',
        'NumericRating': 'rating'
    })
    
    # Traitement basique des valeurs manquantes
    df_work['price'] = df_work['price'].fillna(0)
    df_work['purchases'] = df_work['purchases'].fillna(0)
    df_work['rating'] = df_work['rating'].fillna(0)
    
    # Garder seulement les lignes avec des noms d'apps valides
    df_work = df_work.dropna(subset=['app_name'])
    df_work = df_work[df_work['app_name'].str.len() > 0]
    
    # Supprimer les doublons basés sur le nom de l'app
    df_work = df_work.drop_duplicates(subset=['app_name'], keep='first')
    
    print(f"✅ Données préparées: {len(df_work)} applications")
    print("\n" + "="*50 + "\n")
    return df_work

def create_performance_clusters(df):
    """Création des clusters basés sur les performances"""
    print("🎯 CRÉATION DES CLUSTERS DE PERFORMANCE")
    print("="*50)
    
    # Calcul du revenue estimé
    df['revenue_estimate'] = df['price'] * df['purchases']
    
    # Calcul des percentiles pour définir les seuils
    purchase_percentiles = df['purchases'].quantile([0.25, 0.50, 0.75, 0.90]).to_dict()
    revenue_percentiles = df['revenue_estimate'].quantile([0.25, 0.50, 0.75, 0.90]).to_dict()
    price_percentiles = df['price'].quantile([0.25, 0.50, 0.75]).to_dict()
    rating_median = df['rating'].median()
    
    print("📊 Seuils de classification:")
    print(f"Achats - P25: {purchase_percentiles[0.25]:.0f}, P50: {purchase_percentiles[0.50]:.0f}, P75: {purchase_percentiles[0.75]:.0f}, P90: {purchase_percentiles[0.90]:.0f}")
    print(f"Revenue - P25: {revenue_percentiles[0.25]:.0f}, P50: {revenue_percentiles[0.50]:.0f}, P75: {revenue_percentiles[0.75]:.0f}, P90: {revenue_percentiles[0.90]:.0f}")
    print(f"Prix - P25: {price_percentiles[0.25]:.2f}, P50: {price_percentiles[0.50]:.2f}, P75: {price_percentiles[0.75]:.2f}")
    print(f"Note médiane: {rating_median:.2f}")

    def classify_app(row):
        # 1. High Performers (Top 10% revenue et achats élevés)
        if (row['revenue_estimate'] >= revenue_percentiles[0.90] and 
            row['purchases'] >= purchase_percentiles[0.75]):
            return "High Performers"
        
        # 2. Premium (Prix élevé, revenue correct)
        elif (row['price'] >= price_percentiles[0.75] and 
              row['revenue_estimate'] >= revenue_percentiles[0.50]):
            return "Premium"
        
        # 3. Mass Market (Beaucoup d'achats, prix bas)
        elif (row['purchases'] >= purchase_percentiles[0.75] and 
              row['price'] <= price_percentiles[0.25]):
            return "Mass Market"
        
        # 4. Growing (Revenue moyen, bonne note)
        elif (row['revenue_estimate'] >= revenue_percentiles[0.50] and 
              row['rating'] >= rating_median):
            return "Growing"
        
        # 5. Low Impact (Le reste)
        else:
            return "Low Impact"
    
    df['cluster'] = df.apply(classify_app, axis=1)
    
    # Statistiques des clusters
    print("\n📈 DISTRIBUTION DES CLUSTERS:")
    cluster_stats = df['cluster'].value_counts().sort_values(ascending=False)
    for name, count in cluster_stats.items():
        print(f"{name}: {count} apps ({count/len(df)*100:.1f}%)")
    
    # Analyse détaillée par cluster
    print("\n📊 PROFIL DE CHAQUE CLUSTER:")
    for cluster in df['cluster'].unique():
        cluster_data = df[df['cluster'] == cluster]
        print(f"\n{cluster} ({len(cluster_data)} apps):")
        print(f"  💰 Prix moyen: {cluster_data['price'].mean():.2f}€ (médiane: {cluster_data['price'].median():.2f}€)")
        print(f"  🛍️ Achats moyens: {cluster_data['purchases'].mean():.0f} (médiane: {cluster_data['purchases'].median():.0f})")
        print(f"  💵 Revenue moyen: {cluster_data['revenue_estimate'].mean():.0f}€ (médiane: {cluster_data['revenue_estimate'].median():.0f}€)")
        print(f"  ⭐ Note moyenne: {cluster_data['rating'].mean():.2f} (médiane: {cluster_data['rating'].median():.2f})")
    
    print("\n" + "="*50 + "\n")
    return df

def create_cluster_visualizations(df, output_dir):
    """Création des graphiques d'analyse des clusters"""
    print("📊 CRÉATION DES VISUALISATIONS")
    print("="*50)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Palette de couleurs pour les clusters
    colors = {
        "High Performers": "#2ECC71",    # Vert
        "Premium": "#9B59B6",           # Violet
        "Mass Market": "#3498DB",       # Bleu
        "Growing": "#F39C12",           # Orange
        "Low Impact": "#95A5A6"         # Gris
    }
    
    # 1. Scatter plot Revenue vs Achats
    plt.figure(figsize=(12, 8))
    for cluster in df['cluster'].unique():
        cluster_data = df[df['cluster'] == cluster]
        plt.scatter(cluster_data['purchases'], 
                   cluster_data['revenue_estimate'],
                   c=colors.get(cluster, '#34495E'),
                   label=cluster,
                   alpha=0.7,
                   s=60)
    
    plt.xlabel('Nombre d\'achats')
    plt.ylabel('Revenue estimé (€)')
    plt.title('Revenue vs Achats par Cluster')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Log scale si les valeurs sont très dispersées
    if df['purchases'].max() > 1000:
        plt.xscale('log')
    if df['revenue_estimate'].max() > 10000:
        plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'clusters_revenue_vs_purchases.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Distribution des clusters (Camembert)
    plt.figure(figsize=(10, 8))
    cluster_counts = df['cluster'].value_counts()
    colors_list = [colors.get(cluster, '#34495E') for cluster in cluster_counts.index]
    
    plt.pie(cluster_counts.values, 
            labels=cluster_counts.index,
            colors=colors_list,
            autopct='%1.1f%%',
            startangle=90)
    plt.title('Distribution des Applications par Cluster', fontsize=16)
    plt.savefig(os.path.join(output_dir, 'clusters_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Comparaison des métriques moyennes
    plt.figure(figsize=(15, 10))
    
    metrics = ['price', 'purchases', 'revenue_estimate', 'rating']
    metric_names = ['Prix moyen (€)', 'Achats moyens', 'Revenue moyen (€)', 'Note moyenne']
    
    cluster_means = df.groupby('cluster')[metrics].mean()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        ax = axes[i]
        bars = ax.bar(cluster_means.index, cluster_means[metric], 
                     color=[colors.get(cluster, '#34495E') for cluster in cluster_means.index])
        ax.set_title(name, fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        
        # Ajouter les valeurs sur les barres
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'clusters_metrics_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Visualisations sauvegardées dans {output_dir}/")
    print("\n" + "="*50 + "\n")

def generate_detailed_report(df, output_dir):
    """Génération d'un rapport détaillé"""
    print("📝 GÉNÉRATION DU RAPPORT DÉTAILLÉ")
    print("="*50)
    
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, 'odoo_cluster_analysis_report.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 📊 Analyse des Clusters - Applications Odoo\n\n")
        f.write(f"**Date d'analyse:** {datetime.now().strftime('%d/%m/%Y à %H:%M')}\n")
        f.write(f"**Nombre d'applications analysées:** {len(df)}\n\n")
        
        # Vue d'ensemble
        f.write("## 🎯 Vue d'ensemble\n\n")
        f.write("Cette analyse classe les applications Odoo en 5 clusters basés sur leurs performances commerciales :\n\n")
        
        cluster_descriptions = {
            "High Performers": "Applications with very high revenue and many purchases",
            "Premium": "High-priced applications with solid revenue", 
            "Mass Market": "Low-priced applications with high purchase volume",
            "Growing": "Applications with average revenue and good ratings",
            "Low Impact": "Applications with standard performance"
        }
        
        for cluster, desc in cluster_descriptions.items():
            f.write(f"- **{cluster}**: {desc}\n")
        
        # Distribution des clusters
        f.write("\n## 📈 Distribution des Clusters\n\n")
        cluster_counts = df['cluster'].value_counts()
        f.write("| Cluster | Nombre d'apps | Pourcentage |\n")
        f.write("|---------|---------------|-------------|\n")
        for cluster, count in cluster_counts.items():
            percentage = count/len(df)*100
            f.write(f"| {cluster} | {count} | {percentage:.1f}% |\n")
        
        # Statistiques détaillées par cluster
        f.write("\n## 📊 Statistiques Détaillées par Cluster\n\n")
        
        cluster_stats = df.groupby('cluster').agg({
            'price': ['mean', 'median', 'min', 'max'],
            'purchases': ['mean', 'median', 'min', 'max'], 
            'revenue_estimate': ['mean', 'median', 'min', 'max'],
            'rating': ['mean', 'median', 'min', 'max']
        }).round(2)
        
        for cluster in df['cluster'].unique():
            f.write(f"\n### {cluster}\n\n")
            cluster_data = df[df['cluster'] == cluster]
            f.write(f"**Nombre d'applications:** {len(cluster_data)}\n\n")
            
            f.write("| Métrique | Moyenne | Médiane | Min | Max |\n")
            f.write("|----------|---------|---------|-----|-----|\n")
            
            metrics = [('price', 'Prix (€)'), ('purchases', 'Achats'), ('revenue_estimate', 'Revenue (€)'), ('rating', 'Note')]
            for metric, label in metrics:
                stats = cluster_stats.loc[cluster, metric]
                f.write(f"| {label} | {stats['mean']:.2f} | {stats['median']:.2f} | {stats['min']:.2f} | {stats['max']:.2f} |\n")
        
        # Top applications par cluster
        f.write("\n## 🏆 Top 5 Applications par Cluster\n\n")
        for cluster in df['cluster'].unique():
            f.write(f"\n### {cluster}\n\n")
            top_apps = df[df['cluster'] == cluster].nlargest(5, 'revenue_estimate')
            
            if len(top_apps) > 0:
                f.write("| Application | Prix (€) | Achats | Revenue (€) | Note |\n")
                f.write("|-------------|----------|--------|-------------|------|\n")
                for _, app in top_apps.iterrows():
                    f.write(f"| {app['app_name'][:30]}{'...' if len(app['app_name']) > 30 else ''} | {app['price']:.2f} | {app['purchases']:.0f} | {app['revenue_estimate']:.0f} | {app['rating']:.2f} |\n")
            else:
                f.write("*Aucune application dans ce cluster*\n")
        
        # Recommandations stratégiques
        f.write("\n## 💡 Recommandations Stratégiques\n\n")
        recommendations = {
            "High Performers": [
                "Invest heavily in new feature development",
                "Develop premium versions with higher pricing",
                "Use these apps as showcase to attract new developers"
            ],
            "Premium": [
                "Maintain quality and exclusivity",
                "Develop premium packages and bundles",
                "Target niche markets with high added value"
            ],
            "Mass Market": [
                "Optimize development and maintenance costs",
                "Offer freemium versions to increase adoption",
                "Develop cross-selling strategies"
            ],
            "Growing": [
                "Invest in marketing to increase visibility",
                "Improve features based on user feedback",
                "High growth potential to exploit"
            ],
            "Low Impact": [
                "Analyze causes of poor performance",
                "Consider redesign or repositioning",
                "Evaluate profitability and consider discontinuation if necessary"
            ]
        }
        
        for cluster, recs in recommendations.items():
            f.write(f"\n### {cluster}\n")
            for rec in recs:
                f.write(f"- {rec}\n")
        
        f.write(f"\n---\n*Rapport généré automatiquement le {datetime.now().strftime('%d/%m/%Y à %H:%M')}*\n")
    
    print(f"✅ Rapport détaillé sauvegardé: {report_path}")
    print("\n" + "="*50 + "\n")

def main():
    """Fonction principale d'analyse"""
    print("🚀 DÉMARRAGE DE L'ANALYSE DE CLUSTERING ODOO")
    print("="*60)
    
    # Chargement des données
    try:
        df = pd.read_csv('odoo.apps.csv')
        print(f"✅ Fichier chargé avec succès: {len(df)} applications")
    except FileNotFoundError:
        print("❌ Fichier 'odoo.apps.csv' non trouvé dans le répertoire courant")
        return
    except Exception as e:
        print(f"❌ Erreur lors du chargement: {e}")
        return
    
    # Aperçu des données
    df = basic_data_overview(df)
    
    # Préparation des données
    df_prepared = prepare_data_for_clustering(df)
    
    # Clustering
    df_clustered = create_performance_clusters(df_prepared)
    
    # Création du dossier de sortie
    output_dir = 'odoo_cluster_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Sauvegarde des données avec clusters
    output_file = os.path.join(output_dir, 'odoo_apps_with_clusters.csv')
    df_clustered.to_csv(output_file, index=False, encoding='utf-8')
    print(f"💾 Données avec clusters sauvegardées: {output_file}")
    
    # Génération des visualisations
    create_cluster_visualizations(df_clustered, output_dir)
    
    # Génération du rapport
    generate_detailed_report(df_clustered, output_dir)
    
    print("🎉 ANALYSE TERMINÉE AVEC SUCCÈS!")
    print(f"📁 Tous les résultats sont disponibles dans: {os.path.abspath(output_dir)}")
    print("\nFichiers générés:")
    print("- odoo_apps_with_clusters.csv (données avec clusters)")
    print("- clusters_revenue_vs_purchases.png (graphique scatter)")
    print("- clusters_distribution.png (camembert)")
    print("- clusters_metrics_comparison.png (comparaison des métriques)")
    print("- odoo_cluster_analysis_report.md (rapport détaillé)")

if __name__ == "__main__":
    main()