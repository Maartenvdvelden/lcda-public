import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, make_scorer, confusion_matrix, classification_report

import itertools


#Layout figuren:
plt.rcParams['figure.facecolor'] = 'w' #kleur om plot heen
plt.rcParams['axes.facecolor'] = 'w' #kleur in plot
#Grid
plt.rcParams['axes.grid'] = True #verwijder grid indien False
plt.rcParams['axes.grid.axis'] = 'y' #indien axes.grid = True, toon alleen y-as grid
plt.rcParams['grid.color'] = 'k' #indien axes.grid = True
plt.rcParams['grid.alpha'] = 0.1 #indien axes.grid = True
#Frame
plt.rcParams['axes.edgecolor'] = '#23114c' #Opaque Caché
plt.rcParams['axes.linewidth'] = 1.5 #dikte frame
plt.rcParams['axes.spines.top'] = False #Verwijder bovenkant frame
plt.rcParams['axes.spines.right'] = False #Verwijder rechterkant frame
#Ticks
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
#Labels
plt.rcParams['xtick.labelsize'] = 'x-large'
plt.rcParams['xtick.color'] = '#23114c' #Opaque Caché
plt.rcParams['ytick.labelsize'] = 'x-large'
plt.rcParams['ytick.color'] = '#23114c' #Opaque Caché
plt.rcParams['legend.fontsize'] = 'xx-large'
plt.rcParams['legend.title_fontsize'] = 'xx-large'
plt.rcParams['axes.titlesize'] = 'xx-large'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.titlecolor'] = '#20BDBA' #Turquoise
plt.rcParams['axes.labelsize'] = 'xx-large'
plt.rcParams['axes.labelcolor'] = '#23114c' #Opaque Caché
plt.rcParams['text.color'] = '#23114c' #Opaque Caché
#Font
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Verdana'] #of Source Sans Pro?
#Lijnen
plt.rcParams['lines.linewidth'] = 3 #lijnen in de plot

#parameters figuren functies
height = 5.5
titlesize = 21
fontsize = 13
titelhoogte = 1.08
ytext = 10

def koppel_1cho_stnr():
    """Leest 1cijferHO bestand in voor kolom gem. cijfer vooropleiding,
    koppelt BSN aan studentnummer en returns tabal met stnr en gem cijfer
    
    Returns:
        dataframe: pandas dataframe met kolommen: 'Studentnummer' en 'VooropleidingGemCijfer'
    """
    
    # lees 1cijferHO in
    df = pd.read_parquet(r'F:\PermanentStorage\1CijferHO\2021-2022\versie 2\1CIJFERHO_21RI-22_VERSIE2\unsafe\EV21RI22.parquet')
    # selecteer gewenste kolommen
    df = df[['Burgerservicenummer', 'GemEindcijferVoVanDeHoogsteVooroplVoorHetHo']].copy()
    # drop dubbele studenten, bewaar de meest recente (een paar studenten hebben meerdere verschillende 'GemEindcijferVoVanDeHoogsteVooroplVoorHetHo')
    df.drop_duplicates(subset = 'Burgerservicenummer', keep = 'last', inplace = True)
    # hernoem kolom
    df.rename(columns = {'GemEindcijferVoVanDeHoogsteVooroplVoorHetHo': 'VooropleidingGemCijfer'}, inplace = True)
    
    # lees tabel met studentnummer en BSN in
    df_stnr_bsn = pd.read_excel(r'F:\AnalyticsData\maarten-lcda\unsafe\stnr-bsn.xlsx', converters= {'Studentnummer':str, 'Burgerservicenummer':str})
    
    # koppel 1cijferHO en tabel met studentnummer en BSN    
    data = df.merge(df_stnr_bsn, how = 'left', on = 'Burgerservicenummer')
    # controleer of resulterende data dezelfde lengte heeft als initiele data
    assert len(data) == len(df), 'mergen zorgt voor extra regels!'
    # drop rijen zonder studentnummer (veel bsn's hebben geen studennummer (1cijferHO gaat verder terug in het verleden dan Osiris))
    data.dropna(subset = ['Studentnummer'], inplace = True)
    # drop BSN
    data = data[['Studentnummer', 'VooropleidingGemCijfer']].copy()
    
    return data

def bewerk_osiris_data(df_stnr_gemcijfer):
    """Leest oktoberinschrijvingen in en voegt berekende kolommen toe met
    leeftijd op 1 okt in het collegejaar. Merged data met 1cijferHO data 
    met studentnummer en gem cijfer vooropleiding.
    
    Args:
        df_stnr_gemcijfer: pandas dataframe uit functie koppel_1cho_stnr.
        
    Returns:
        dataframe: pandas dataframe. 
    
    """
    
    # dataset inladen met oktoberinschrijvingen
    # verder bevat deze data extra kolommen als postcode die mogelijk interessant zijn
    df = pd.read_feather(r'F:\AnalyticsData\maarten-lcda\oktober_inschrijving.feather') #versie 2022-04-20
    
    # voeg kolom toe met de datum van 1 oktober in betreffende collegejaar
    df["1okt"] = df.Collegejaar.astype(str) + "-10-01"
    # zet kolom om in datetime
    df["1okt"] = pd.to_datetime(df["1okt"])
    # voeg berekende kolom met leeftijd op 1 okt toe en maak er integers van
    df["Leeftijd"] = (
        (df["1okt"] - df["Geboortedatum"])
        .div(np.timedelta64(1, "Y"))
        .astype("int64")
    )    
    # selecteer alleen studenten met een 'hoofdopleiding', bekostiging isin J, N (gebeurd ook in studievoortgang (sv) data)
    df = df[df.Bekostiging.isin(['J', 'N'])].copy()    
    # check of er studenten zijn die alsnog binnen een collegejaar meerdere inschrijvingen bij dezelfde opleiding hebben 
    # dit is één student: df[df.duplicated(subset = ['Studentnummer', 'OpleidingCode', 'Collegejaar']]
    # bewaar de D inschrijving (binnen sept is deze student met D en B gestart)
    df = df.drop_duplicates(subset = ['Studentnummer', 'OpleidingCode', 'Collegejaar'], keep = 'last')
    # let op! er zijn alsnog 2 studenten met inschrijvingen bij meerdere opleidingen: wordt later bij mergen sv data gecorrigeerd
    
    # selecteer gewenste kolommen om met studievoortgang data te mergen (er zijn studenten die vaker in een jaar voorkomen, dus neem OpleidingCode mee)
    # voorkom kolommen met dezelfde naam, want dan krijg je _x en _y indien ze niet in de merge voorkomen
    df = df[['Studentnummer', 'Leeftijd', 'OpleidingCode', 'Collegejaar', 'DatumVerzoekInschrijving', 'Ingangsdatum', 'VooropleidingBRIN4', 'VooropleidingPC4']].copy()
    
    #merge met 1cijferHO data met gem cijfer vooropleiding
    data = df.merge(df_stnr_gemcijfer, how = 'left', on = 'Studentnummer')
    # controleer of resulterende data dezelfde lengte heeft als initiele data
    assert len(data) == len(df), 'mergen zorgt voor extra regels!'
    
    return data

def koppel_sv_osiris(df_osiris):
    """Leest studievoortgangsdata in en koppelt aan Osirisdata.
    
    Args:
        df_osiris: pandas dataframe uit functie bewerk_osiris_data.
    
    Returns:
        dataframe: pandas dataframe.
    """
    
    # dataset die HL voor studievoortgang monitoring gebruikt inladen
    df = pd.read_feather(r'F:\Hogeschool Leiden\IR-data - data met persoonsgegevens\2021-2022\monitoring-impact-corona\raw\2022-04-20\sv_student_collegejaar_periode_MET_STUDENTNUMMER.feather')
    # bewaar alle rijen in df_mic en voeg informatie uit df_osiris toe
    data = df.merge(df_osiris, how = 'left', on = ['Studentnummer', 'OpleidingCode', 'Collegejaar'])
    # controleer of resulterende data dezelfde lengte heeft als initiele data
    assert len(data) == len(df), 'mergen zorgt voor extra regels!'
   
    return data

def data_bewerking(df_sv):
    """Filter gewenste data, voeg (extra) berekende kolommen toe, zoals
    verblijfsjaaropleiding (niet gecorrigeeerd voor tussenjaren etc., 
    want is mogelijk moeilijk bij andere instellingen te reproduceren), 
    langstudeerder en of een student langstudeerdder wordt.
    Hernoem kolommen, selecteer 1 student per collegejaar (ipv 5), 
    selecteer gewenste eindkolommen.
    
    Args:
        df_sv: pandas dataframe uit functie koppel_sv_osiris.
    
    Returns:
        dataframe: pandas dataframe.
    """
    
    # dataset
    df = df_sv.copy()
    # selecteer voltijd bachelor studenten
    # Let op! In het verleden startten Ad studenten in de D-fase in hun eerste jaar (nu niet meer),
    # dus verwijder de Ad opleidingen Management in de Zorg, Onderwijsondersteuner Gezondheidszorg 
    # en Welzijn Onderwijsondersteuner Omgangskunde: 80011, 80045, 80060
    filters = (df.CrohoActueel.isin(['80011', '80045', '80060']) == False) & \
              (df.Opleidingsvorm == 'voltijd') & \
              (df.ExamentypeCSA.isin(['D', 'B']))
    df = df[filters].copy()    
    
    # maak een (tijdelijke) unieke identifier van Studentnummer + OpleidingCode, zodat
    # lijsten maken van studenten per opleiding alleen invloed heeft op specifieke studenten
    # in een specifieke opleiding en niet op die student bij alle opleidingen.
    # als iemand langstudeerder wordt (zie hieronder) bij opleiding A, maar niet bij opleiding B,
    # dan wil je dat dit alleen bij opleiding A gemarkeerd wordt
    df['UniekID'] = df.Studentnummer + df.OpleidingCode    
    # Vervang bovenstaand UniekID door 'categorie' identifier
    # Let op! Student X die bij opleiding A studeert en in een ander jaar bij opleiding B heeft 
    # dus 2 ID's en wordt dus als 2 studenten gezien
    df['ID'] = df.UniekID.astype('category').cat.codes    
    # sorteer op ID en leeftijd
    df.sort_values(by = ['ID', 'Leeftijd'], inplace = True)
    
    # voeg berekende kolommen toe
    # verschil in dagen tussen Ingangsdatum en DatumVerzoekInschrijving
    df['DagenTotIngangsdatum'] = (df.Ingangsdatum - df.DatumVerzoekInschrijving).dt.days
    # propedeuse binnen 1 jaar gehaald?
    df['Pin1'] = np.where(df.CollegejaarPropedeuseDiploma == df.AanvangsjaarOpleiding, 1, 0)
    # diploma binnen 4 jaar gehaald? Wordt dus geen langstudeerder
    df['Din4'] = np.where(df.CollegejaarDiploma <= df.AanvangsjaarOpleiding + 3, 1, 0)    
    # diploma binnen 5 jaar gehaald? Wordt dus geen langstudeerder
    df['Din5'] = np.where(df.CollegejaarDiploma <= df.AanvangsjaarOpleiding + 4, 1, 0)
    # diploma in jaar X gehaald
    df['DinJaarX'] = df.CollegejaarDiploma - df.AanvangsjaarOpleiding + 1
    # maak een kolom met ongecorrigeerd Verblijfsjaar
    df['VerblijfsjaarOpleiding'] = df.Collegejaar - df.AanvangsjaarOpleiding + 1  
    
    # maak df met ID en aanvangsdatum opleiding 
    df_datum = df.groupby('ID').Ingangsdatum.min().reset_index(name = 'AanvangsdatumOpleiding')
    # merge met df
    voor_merge = len(df)
    df = df.merge(df_datum, how = 'left', on = ['ID'])
    na_merge = len(df)
    assert voor_merge == voor_merge, 'mergen zorgt voor extra regels!' 
    # maanden tot diploma
    df['MaandentotD'] = (df.DatumDiploma - df.AanvangsdatumOpleiding) / np.timedelta64(1, 'M')
    # maak kolommen voor langstudeerders (of een student in het betreffende jaar langstudeerder is)
    #start aan 5e of 6e jaar
    df['Langstudeerder5'] = np.where(df.VerblijfsjaarOpleiding >= 5, 1, 0)
    df['Langstudeerder6'] = np.where(df.VerblijfsjaarOpleiding >= 6, 1, 0)
    # maak lijsten van studenten die langstudeerder zijn of worden
    lijst_lang5 = df[df.Langstudeerder5 == 1].ID.unique()
    lijst_lang6 = df[df.Langstudeerder6 == 1].ID.unique()
    # voeg kolommen toe met of student langstudeerder wordt
    # een student wordt zeker geen langstudeerder als die Din4 of Din5 jaar haalt (afh van definitie),
    # indien student nog geen langstudeerder is en geen Din4 of Din5, dan is het nog
    # onbekend of die langstudeerder wordt
    # wordt langstudeerder door start aan jaar 5 of juist niet door Diploma in 4 jaar
    conditions = [
        (df.ID.isin(lijst_lang5)),
        (df.Din4 == 1),
    ]                
    
    choices = [
        1,
        0
    ]
    
    df['WordtLangstudeerder5'] = np.select(conditions, choices, default = None)
    
    #wordt langstudeerder door start aan jaar 6 of juist niet door Diploma in 5 jaar
    conditions = [
        (df.ID.isin(lijst_lang6)),
        (df.Din5 == 1),
    ]                
    
    choices = [
        1,
        0
    ]
    
    df['WordtLangstudeerder6'] = np.select(conditions, choices, default = None)
    
    # EXTRA DF VOOR VISUALISEREN LANGSTUDEERDERS #
    # maak dataframe met alle verblijfsjaren per collegejaar om langstudeerders te plotten
    df_ruw = df.copy()
    # bewaar alleen 1 regel per student per collegejaar (selecteer dus 1 periode, zou niet uit moeten maken welke)
    df_ruw = df_ruw[df_ruw.PeriodeNummer == 1].copy()
    # EXTRA DF VOOR VISUALISEREN LANGSTUDEERDERS #
    
    # maak lijst met studenten die in de p-fase instromen
    # let op! deze selectie zorgt ervoor dat enkel studenten die vanaf 2012 starten worden meegenomen
    # (omdat 2012 het eerst meegenomen collegejaar is).
    # met deze data kunnen dus geen grafieken van % langstudeerders per jaar worden gemaakt.
    lijst_p = df[(df.VerblijfsjaarOpleiding == 1) & (df.ExamentypeCSA == 'D')].ID.unique()
    # selecteer alleen studenten die in de P-fase instromen
    df = df[df.ID.isin(lijst_p)].copy()   
    
    # vervang PeriodeNummer door tmP1 etc. en maak er string van, want bevat ECs tot en met bepaalde periode
    df['PeriodeNummer'] = 'tmP' + df.PeriodeNummer.astype('str')    
    # hernoem kolommen
    df.rename(columns = {'BehaaldeECcumulatief': 'EC', 'BehaaldeECGeenVrijstellingcumulatief': 'ECGeenVrij'}, inplace = True)
    
    # maak df met studiepunten (EC) om te mergen met df
    # bij index kun je helaas niet alle kolommen opgeven, want NaNs kunnen niet in index komen, dus zouden afvallen
    df_ec = df.pivot_table(values = ['EC', 'ECGeenVrij'], fill_value = 0, columns = 'PeriodeNummer', index = ['ID', 'Collegejaar'])
    # fuseer kolomnamen multiindex
    df_ec.columns = df_ec.columns.map(''.join).str.strip('') # geen idee of .str.strip('') nodig is
    # verwijder multiindex
    df_ec = df_ec.reset_index()
    # merge df and df_ec
    data = df.merge(df_ec, how = 'left', on = ['ID', 'Collegejaar'])
    # controleer of resulterende data dezelfde lengte heeft als initiele data
    assert len(data) == len(df), 'mergen zorgt voor extra regels!'  
    # bewaar alleen 1 regel per student per collegejaar (selecteer dus 1 periode, zou niet uit moeten maken welke)
    data = data[data.PeriodeNummer == 'tmP1'].copy()
    # reset index
    data = data.reset_index(drop = True)
    
    # selecteer uiteindelijke kolommen (in volgorde naar wens)
    data = data[['ID',
    'Geslacht',
    'Leeftijd',
    'Vooropleiding',
    'VooropleidingBRIN4',
    'VooropleidingPC4',
    'VooropleidingGemCijfer',
    'DagenTotIngangsdatum',
    'AanvangsjaarOpleiding',
    'Collegejaar',
    'VerblijfsjaarOpleiding',
    'Langstudeerder5',
    'Langstudeerder6',
    'WordtLangstudeerder5',
    'WordtLangstudeerder6',
    'CrohoActueel',
    'NaamOpleiding',
    'Faculteit',
    'StatusJaarLater',
    'Pin1',
    'Din4',
    'Din5',
    'DinJaarX',
    'MaandentotD',
    'ECtmP1',
    'ECtmP2',
    'ECtmP3',
    'ECtmP4',
    'ECtmP5',
    'ECGeenVrijtmP1',
    'ECGeenVrijtmP2',
    'ECGeenVrijtmP3',
    'ECGeenVrijtmP4',
    'ECGeenVrijtmP5']].copy()
    
    return df_ruw, data

def plot_langstudeerders(data, titel = 'Percentage langstudeerders HL (voltijd)'):
    """ Maakt plot van het het percentage langstudeerders van hogeschool leiden.
    
    Args:
        data: df met langstudeerders (%), zoals df_lang
        titel: string met titel, default is 'Percentage langstudeerders HL (voltijd)'
    
    Returns g: seaborn plot
    """
   
    g = sns.relplot(x = 'Collegejaar', y = 'Percentage', data = data, legend = False, kind = 'line', ci = None, palette = ['#20BDBA'], height = height, aspect = 1.5)
    g.set(xlabel = 'Academic year', ylabel = 'Percentage (%)')
    g.set(ylim = (0, (data.Percentage.max()) * 1.1))
    g.set(xticks = data.Collegejaar.values) #OF   sorted(df.Collegejaar.unique()) ?
    g.fig.suptitle(titel, y = titelhoogte, fontsize = titlesize, fontweight = 'bold', color = '#20BDBA', horizontalalignment = 'center')
        
    return g

def plot_cat_kenmerken(data, kenmerk, titel = 'Kans op langstuderen HL (voltijd)'):
    """ Maakt plot van het percentage eerstejaars tussen 2012-2016 die
    langstudeerder worden van hogeschool leiden naar kenmerk. Dus geeft de kans 
    op langstuderen weer.
    
    Args:
        data: df met langstudeerders (%), zoals df_kenmerk
        kenmerk: categorisch kenmerk (kolomnaam)
        titel: string met titel, default is 'Percentage langstudeerders'
    
    Returns g: seaborn plot
    """
   
    g = sns.catplot(x = 'Percentage', y = kenmerk, data = data, kind = 'bar', ci = None, palette = ['#20BDBA', '#008385', '#23114c', '#000000', '#424747', '#B2B1B1'], height = height, aspect = 1.5)
    g.set(xlabel = '', ylabel = '') # verwijder aslabels 
    g.fig.suptitle(titel, x = 0.15, y = 1.22, fontsize = titlesize, fontweight = 'bold', color = '#20BDBA', horizontalalignment = 'left')

    return g

def plot_num_kenmerken(data, kenmerk, feature, titel = 'Gemiddelde leeftijd HL (voltijd)'):
    """ Maakt plot van het gemiddelde van het kenmerk van eerstejaars tussen 2012-2016 die
    langstudeerder worden van hogeschool leiden.
    
    Args:
        data: df met gemiddelde per kenmerk uitgesplitst naar wel of niet langstuderen, zoals df_kenmerk
        kenmerk: categorisch kenmerk (kolomnaam)
        feature: categorisch kenmerk engelstalig
        titel: string met titel, default is 'Percentage langstudeerders'
    
    Returns g: seaborn plot
    """
   
    g = sns.catplot(x = 'WordtLangstudeerder6', y = kenmerk, data = data, kind = 'bar', ci = None, palette = ['#20BDBA', '#008385'], height = height, aspect = 1.5)
    g.set(xlabel = 'Wordt langstudeerder', ylabel = 'Gemiddelde') 
    g.fig.suptitle(titel, y = titelhoogte, fontsize = titlesize, fontweight = 'bold', color = '#20BDBA', horizontalalignment = 'center')

    return g

# deze functie komt uit een IBM data science studie van edX.org
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.rcParams['axes.grid'] = False #verwijder grid indien False
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Observed')
    plt.xlabel('Predicted')
    
    plt.rcParams['axes.grid'] = True #voeg grid weer toe voor andere figuren
    
    
def model(df_kenmerken, df_label, est, name_est):
    """Functie om modellen te testen, kenmerken en plotjes te printen"""
    
    print("Estimator", name_est)
    
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(df_kenmerken, 
                                                        df_label.values.ravel(), 
                                                        test_size = 0.2, 
                                                        #stratify = df_label.values.ravel(), 
                                                        random_state = 14)

    # model fitten aan train data
    est.fit(X_train, y_train)
    
    # train data voorspellen
    train_pred = est.predict(X_train)
    
    # y voorspellen op basis van X_test -> het beste model geeft y_test waarden terug
    y_pred = est.predict(X_test)
    
    #EXTRA zelf threshold zetten:
    # y_pred = (est.predict_proba(X_test)[:,1] >= 0.2).astype(bool)
    
    
    print("waargenomen", y_test[0:20])
    print("voorspelling", y_pred[0:20])
    
    
    
    # kans op y = 0 en y = 1 voorspellen op basis van X_test
    y_pred_prob = est.predict_proba(X_test)[:,1]
    print("probability", np.round_(y_pred_prob[0:20], decimals = 2))
    
    # plot roc
    fig, ax = plt.subplots(1, 1, figsize = (5, 5))
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], color ='k', linestyle = '--')
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_title(f"Study delay AUC = {roc_auc_score(y_test, y_pred_prob):.4f}")
    
    plt.show()
        
    # Accuracy -> hoeveel % juist voorspeld?    
    print("Accuracy score train:", accuracy_score(y_train, train_pred))
    print("Accuracy score test:", accuracy_score(y_test, y_pred))
    print("Roc auc score (op y_pred_proba):", roc_auc_score(y_test, y_pred_prob))
    print("Roc auc score (op y_pred):", roc_auc_score(y_test, y_pred))
    
    # confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred, labels = [1, 0])
    np.set_printoptions(precision=2)
    plot_confusion_matrix(cnf_matrix, classes=['delayed','not delayed'] ,normalize=False,  title='Confusion matrix')
    plt.show()
    
    # classification report
    print (classification_report(y_test, y_pred))
    
    
    print("\n")
    
def cross_val(df_kenmerken, df_label, est, name_est):
    """Functie om data in willekeurige delen op te hakken en modellen te testen"""
    
    print("Estimator", name_est)
    
    k = 5
    kf = KFold(n_splits = 5, shuffle = True, random_state = 14)
    
    acc_score = []
    roc_auc_score_list = []
    
    for train_index, test_index in kf.split(df_kenmerken):
        X_train, X_test = df_kenmerken.iloc[train_index, :], df_kenmerken.iloc[test_index, :]
        y_train, y_test = df_label.iloc[train_index, :].values.ravel(), df_label.iloc[test_index, :].values.ravel()

        est.fit(X_train, y_train)
        y_pred = est.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        acc_score.append(acc)
        
        y_pred_prob = est.predict_proba(X_test)[:,1]
        
        roc_auc = roc_auc_score(y_test, y_pred_prob)
        roc_auc_score_list.append(roc_auc)
        
    avg_acc_score = sum(acc_score)/k
    avg_roc_auc_score = sum(roc_auc_score_list)/k
    
    
    print('accuracy of each fold - {}'.format(acc_score))
    print('Avg accuracy : {}\n'.format(avg_acc_score))
    
    print('roc_auc (y_pred_prob) of each fold - {}'.format(roc_auc_score_list))
    print('Avg roc_auc : {}\n'.format(avg_roc_auc_score))    
    
    print('\n')
    
def cross_val2(df_kenmerken, df_label, est, name_est):
    """Functie om data in willekeurige delen op te hakken en modellen te testen"""
    """print alleen avg accuracy en roc"""

    
    k = 5
    kf = KFold(n_splits = 5, shuffle = True, random_state = 14)
    
    acc_score = []
    roc_auc_score_list = []
    
    for train_index, test_index in kf.split(df_kenmerken):
        X_train, X_test = df_kenmerken.iloc[train_index, :], df_kenmerken.iloc[test_index, :]
        y_train, y_test = df_label.iloc[train_index, :].values.ravel(), df_label.iloc[test_index, :].values.ravel()

        est.fit(X_train, y_train)
        y_pred = est.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        acc_score.append(acc)
        
        y_pred_prob = est.predict_proba(X_test)[:,1]
        
        roc_auc = roc_auc_score(y_test, y_pred_prob)
        roc_auc_score_list.append(roc_auc)
        
    avg_acc_score = sum(acc_score)/k
    avg_roc_auc_score = sum(roc_auc_score_list)/k
    
    
    print('Avg accuracy : {}'.format(avg_acc_score))

    print('Avg roc_auc : {}'.format(avg_roc_auc_score))    
    
    print('\n')
    