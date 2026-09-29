"""Generate the Focon Assistant end-user guide as a self-contained DOCX file."""

from datetime import date
from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "docs" / "Manuel_utilisation_Focon_Assistant.docx"


def xml_text(value):
    return escape(str(value), {"\n": "&#10;"})


def run(text, bold=False, color=None, size=None, italic=False):
    props = []
    if bold:
        props.append("<w:b/>")
    if italic:
        props.append("<w:i/>")
    if color:
        props.append(f'<w:color w:val="{color}"/>')
    if size:
        props.append(f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/>')
    prop_xml = f"<w:rPr>{''.join(props)}</w:rPr>" if props else ""
    preserve = ' xml:space="preserve"' if str(text).startswith(" ") or str(text).endswith(" ") else ""
    return f"<w:r>{prop_xml}<w:t{preserve}>{xml_text(text)}</w:t></w:r>"


def paragraph(text="", style=None, align=None, before=0, after=100, shade=None, border=False):
    ppr = []
    if style:
        ppr.append(f'<w:pStyle w:val="{style}"/>')
    if align:
        ppr.append(f'<w:jc w:val="{align}"/>')
    ppr.append(f'<w:spacing w:before="{before}" w:after="{after}"/>')
    if shade:
        ppr.append(f'<w:shd w:val="clear" w:fill="{shade}"/>')
    if border:
        ppr.append('<w:pBdr><w:bottom w:val="single" w:sz="8" w:space="6" w:color="1F4E79"/></w:pBdr>')
    return f"<w:p><w:pPr>{''.join(ppr)}</w:pPr>{run(text)}</w:p>"


def rich_paragraph(parts, style=None, align=None, after=110, shade=None):
    ppr = []
    if style:
        ppr.append(f'<w:pStyle w:val="{style}"/>')
    if align:
        ppr.append(f'<w:jc w:val="{align}"/>')
    ppr.append(f'<w:spacing w:after="{after}"/>')
    if shade:
        ppr.append(f'<w:shd w:val="clear" w:fill="{shade}"/>')
    body = "".join(run(*part) if isinstance(part, tuple) else run(part) for part in parts)
    return f"<w:p><w:pPr>{''.join(ppr)}</w:pPr>{body}</w:p>"


def bullet(text, level=0):
    left = 420 + level * 360
    return (
        '<w:p><w:pPr><w:pStyle w:val="Normal"/>'
        f'<w:ind w:left="{left}" w:hanging="180"/>'
        '<w:spacing w:after="65"/></w:pPr>'
        f'{run("• ", bold=True, color="1F4E79")}{run(text)}</w:p>'
    )


def numbered(number, text):
    return (
        '<w:p><w:pPr><w:pStyle w:val="Normal"/>'
        '<w:ind w:left="500" w:hanging="240"/><w:spacing w:after="75"/></w:pPr>'
        f'{run(str(number) + ". ", bold=True, color="1F4E79")}{run(text)}</w:p>'
    )


def code_block(text):
    lines = text.split("\n")
    runs = []
    for index, line in enumerate(lines):
        if index:
            runs.append('<w:r><w:br/></w:r>')
        runs.append(run(line, color="17365D"))
    return (
        '<w:p><w:pPr><w:pStyle w:val="Code"/>'
        '<w:shd w:val="clear" w:fill="EAF2F8"/>'
        '<w:spacing w:before="80" w:after="130"/>'
        '</w:pPr>' + "".join(runs) + '</w:p>'
    )


def callout(title, text, color="D9EAF7"):
    return rich_paragraph(
        [(title + " ", True, "17365D", None, False), (text, False, None, None, False)],
        shade=color,
        after=130,
    )


def table(headers, rows, widths=None):
    width_xml = "".join(f'<w:gridCol w:w="{width}"/>' for width in (widths or [2500] * len(headers)))

    def cell(value, header=False):
        fill = "1F4E79" if header else "FFFFFF"
        color = "FFFFFF" if header else "1F1F1F"
        content = run(value, bold=header, color=color)
        return (
            '<w:tc><w:tcPr><w:shd w:val="clear" w:fill="' + fill + '"/>'
            '<w:tcMar><w:top w:w="85" w:type="dxa"/><w:left w:w="100" w:type="dxa"/>'
            '<w:bottom w:w="85" w:type="dxa"/><w:right w:w="100" w:type="dxa"/></w:tcMar>'
            '<w:tcBorders><w:top w:val="single" w:sz="4" w:color="D9E2F3"/>'
            '<w:left w:val="single" w:sz="4" w:color="D9E2F3"/>'
            '<w:bottom w:val="single" w:sz="4" w:color="D9E2F3"/>'
            '<w:right w:val="single" w:sz="4" w:color="D9E2F3"/></w:tcBorders></w:tcPr>'
            '<w:p><w:pPr><w:spacing w:after="0"/></w:pPr>' + content + '</w:p></w:tc>'
        )

    header_row = '<w:tr>' + ''.join(cell(h, header=True) for h in headers) + '</w:tr>'
    data_rows = ''.join('<w:tr>' + ''.join(cell(v) for v in row) + '</w:tr>' for row in rows)
    return (
        '<w:tbl><w:tblPr><w:tblW w:w="0" w:type="auto"/>'
        '<w:tblBorders><w:top w:val="single" w:sz="4" w:color="D9E2F3"/>'
        '<w:left w:val="single" w:sz="4" w:color="D9E2F3"/>'
        '<w:bottom w:val="single" w:sz="4" w:color="D9E2F3"/>'
        '<w:right w:val="single" w:sz="4" w:color="D9E2F3"/>'
        '<w:insideH w:val="single" w:sz="4" w:color="D9E2F3"/>'
        '<w:insideV w:val="single" w:sz="4" w:color="D9E2F3"/></w:tblBorders></w:tblPr>'
        f'<w:tblGrid>{width_xml}</w:tblGrid>{header_row}{data_rows}</w:tbl>'
    )


def heading(text, level=1):
    return paragraph(text, style=f"Heading{level}", before=200 if level == 1 else 140, after=100)


def page_break():
    return '<w:p><w:r><w:br w:type="page"/></w:r></w:p>'


def document_xml():
    parts = []
    parts.append(paragraph("MANUEL D'UTILISATION", style="CoverKicker", align="center", before=900, after=120))
    parts.append(rich_paragraph([("Focon Assistant", True, "17365D", "46", False)], align="center", after=90))
    parts.append(rich_paragraph([("Assistant de trading MT5 avec analyse technique et IA", False, "4F6B7A", "24", False)], align="center", after=300))
    parts.append(paragraph("Version desktop et web - Focon Assistant Trade 1.1", align="center", after=80))
    parts.append(paragraph("Guide pratique pour démarrer, configurer, analyser et valider un signal", align="center", after=520))
    parts.append(callout("Important.", "Focon Assistant aide à analyser et exécuter des ordres vers MetaTrader 5. Il ne garantit ni gain ni performance. Commencez sur un compte de démonstration et gardez la décision finale sous votre contrôle.", "FCE4D6"))
    parts.append(paragraph(f"Document préparé à partir du projet Focon Assistant - {date.today().strftime('%d/%m/%Y')}", align="center", after=0))
    parts.append(page_break())

    parts.append(heading("Sommaire", 1))
    for index, item in enumerate([
        "1. Comprendre Focon Assistant", "2. Préparer son environnement", "3. Démarrer l'application",
        "4. Découvrir les interfaces", "5. Premier scan en mode prudent", "6. Lire un signal",
        "7. Configurer le risque et les marchés", "8. Choisir le mode d'automatisation",
        "9. Utiliser le chat et surveiller les positions", "10. Exemples guidés",
        "11. Résoudre les problèmes fréquents", "12. Glossaire et checklist"
    ], start=1):
        parts.append(numbered(index, item))
    parts.append(callout("Comment lire ce guide.", "Les intitulés de boutons sont reproduits tels qu'ils apparaissent dans l'application. Les montants, prix et signaux des exemples sont fictifs : ils servent uniquement à expliquer la lecture d'un résultat."))

    parts.append(heading("1. Comprendre Focon Assistant", 1))
    parts.append(paragraph("Focon Assistant est un assistant de trading connecté à MetaTrader 5 (MT5). Il récupère les données de marché, recherche des configurations techniques, soumet les résultats à une couche IA et applique des règles de risque avant de proposer ou d'envoyer un ordre."))
    parts.append(heading("Ce que fait le bot", 2))
    for item in [
        "Scanne les symboles visibles dans MT5 ou ceux définis dans la configuration, sur l'unité de temps choisie.",
        "Analyse notamment les EMA 9/21/50/200, RSI, MACD, ATR, volumes, points hauts et bas.",
        "Recherche des éléments Smart Money Concepts : phases AMD, Fair Value Gaps, Break of Structure, prises de liquidité, Order Blocks et setup Boomerang.",
        "Présente une décision BUY, SELL ou WAIT, avec une confiance, un prix d'entrée, un Stop Loss, un Take Profit, un volume et une justification.",
        "Vérifie les conditions d'exécution MT5 et surveille les positions lorsque l'Autopilot est actif."
    ]:
        parts.append(bullet(item))
    parts.append(heading("Chaîne de décision", 2))
    parts.append(table(["Étape", "Rôle"], [
        ["1. Données MT5", "Prix, bougies, compte, positions ouvertes et disponibilité du marché."],
        ["2. Analyse", "Indicateurs techniques, structure de marché, Order Blocks et Boomerang."],
        ["3. IA et filtres", "Confluence, tendance haute unité de temps, anti-FOMO et sens de trading autorisé."],
        ["4. Risque", "Confiance minimale, ratio rendement/risque, taille de lot, pertes et limites de positions."],
        ["5. Action", "Signal affiché, validation manuelle ou exécution automatique selon le mode choisi."]
    ], [1700, 7700]))

    parts.append(heading("2. Préparer son environnement", 1))
    parts.append(heading("Prérequis", 2))
    for item in [
        "Un PC Windows avec MetaTrader 5 installé, ouvert et connecté à un compte de courtage. Pour débuter, utilisez un compte de démonstration.",
        "Le bouton Trading Algorithmique de MT5 doit être activé lorsque vous souhaitez autoriser l'envoi d'ordres.",
        "Les dépendances du projet doivent être installées dans son environnement Python.",
        "Les clés API Gemini et OpenRouter doivent être renseignées dans le fichier .env à la racine du projet. Une clé Groq est optionnelle mais sert de solution de repli rapide.",
        "Ne copiez jamais le contenu du fichier .env dans un chat, une capture publique ou un document partagé."
    ]:
        parts.append(bullet(item))
    parts.append(heading("Vérification avant le premier scan", 2))
    parts.append(table(["À vérifier", "Résultat attendu"], [
        ["MT5", "Le terminal est ouvert et le compte souhaité est connecté."],
        ["Connexion", "L'indicateur MT5 de Focon Assistant affiche connecté."],
        ["Compte", "Le solde, l'équité et la marge libre sont visibles."],
        ["Risque", "Un profil prudent est sélectionné et les limites correspondent au compte."],
        ["Marchés", "Les symboles de l'actif voulu sont visibles dans la liste MT5."]
    ], [2900, 6500]))

    parts.append(heading("3. Démarrer l'application", 1))
    parts.append(paragraph("Ouvrez un terminal dans le dossier du projet, après avoir lancé MT5. Les deux interfaces utilisent la même logique de scan ; choisissez celle qui vous est la plus confortable."))
    parts.append(heading("Interface bureau", 2))
    parts.append(code_block("py main.py\n# ou\npython main.py"))
    parts.append(paragraph("La fenêtre FOCON ASSISTANT TRADE 1.1 s'ouvre. Elle rassemble le scanner, les positions, le chat, les paramètres, le backtest et le module de mémoire évolutive."))
    parts.append(heading("Interface web", 2))
    parts.append(code_block("py web_main.py\n# ou\npython web_main.py\n\nPuis ouvrir : http://127.0.0.1:5000"))
    parts.append(paragraph("Le tableau de bord web s'ouvre dans le navigateur local. Il affiche l'état MT5, le statut de l'Autopilot, le scanner, les positions, la configuration, le compte et le chat."))
    parts.append(callout("À savoir.", "Si les clés Gemini ou OpenRouter manquent, l'application s'arrête avec un message d'erreur. Configurez les clés dans .env, sans les exposer, puis relancez l'application.", "FCE4D6"))

    parts.append(heading("4. Découvrir les interfaces", 1))
    parts.append(heading("Écran bureau", 2))
    parts.append(table(["Zone", "Utilisation"], [
        ["SCANNER LES MARCHÉS", "Lance l'analyse des symboles autorisés sur l'unité de temps configurée."],
        ["AUTOPILOT", "Active ou désactive le cycle automatisé et la surveillance des positions."],
        ["Paramètres", "Règle le profil de risque, les limites, les marchés et le sens BUY/SELL."],
        ["Opportunités (Scanner)", "Affiche les résultats et les boutons PRENDRE POSITION ou IGNORER."],
        ["Positions ouvertes", "Liste ticket, symbole, type, volume, prix, SL, TP et profit."],
        ["Assistant", "Permet de demander une explication à l'IA dans un langage courant."],
        ["BACKTEST MT5", "Ouvre le module de backtest disponible dans la version bureau."],
        ["CONSCIENCE ÉVOLUTIVE", "Affiche mémoire de trades, auto-critique et poids adaptatifs."]
    ], [2800, 6700]))
    parts.append(heading("Tableau de bord web", 2))
    for item in [
        "En haut : statut MT5, état de l'Autopilot, bouton SCANNER LES MARCHÉS, bouton STOP, bouton AUTOPILOT et bouton Positions.",
        "Bloc Scan : progression du scan et cartes de résultats. Un signal négociable propose PRENDRE POSITION ou IGNORER.",
        "Bloc Config : risque par trade, perte journalière maximale, lot maximal, nombre de positions, mode, unité de temps et types de marchés.",
        "Bloc Assistant : saisissez une question et appuyez sur Entrée ou Envoyer.",
        "Bloc Compte : consultation du login, de la balance, de l'équité et de la marge libre."
    ]:
        parts.append(bullet(item))

    parts.append(heading("5. Premier scan en mode prudent", 1))
    parts.append(paragraph("Le premier objectif n'est pas de prendre un trade : il est de vérifier que les données, le filtre de marché et l'interprétation des signaux correspondent à ce que vous voyez dans MT5."))
    steps = [
        "Connectez MT5 à un compte de démonstration et ouvrez Focon Assistant.",
        "Dans les paramètres, sélectionnez le profil Prudent. Dans le projet, ce profil applique une logique de risque plus restrictive que les profils Équilibré ou Dynamique.",
        "Choisissez les types de marchés à autoriser. Par exemple, cochez seulement Crypto si vous souhaitez analyser BTCUSD et éviter les autres classes d'actifs.",
        "Réglez le sens de trading sur Mixte (BUY & SELL) pour une première exploration, ou sur Hausse seule / Baisse seule si votre plan l'impose.",
        "Sélectionnez une unité de temps. Le projet prévoit M5, M15, H1 et H4 ; commencez avec M15 ou H1 pour réduire le bruit du marché.",
        "Cliquez sur SCANNER LES MARCHÉS et attendez le statut Scan terminé. Ne relancez pas un second scan pendant qu'un scan est déjà en cours.",
        "Lisez les résultats. Un WAIT est une réponse normale : il signifie que le bot ne voit pas une opportunité suffisamment claire.",
        "Pour un BUY ou SELL, comparez le symbole, l'entrée, le SL et le TP avec le graphique MT5. Vérifiez également la raison proposée.",
        "Sur compte de démonstration, utilisez PRENDRE POSITION seulement après votre propre validation. Sur la version bureau, une confirmation est demandée avant l'exécution."
    ]
    for index, step in enumerate(steps, start=1):
        parts.append(numbered(index, step))

    parts.append(heading("6. Lire un signal", 1))
    parts.append(paragraph("Un résultat exploitable contient les informations ci-dessous. Le bot peut aussi rendre WAIT, un état volontaire de non-intervention, ou CLOSED lorsque des marchés fermés sont ignorés."))
    parts.append(table(["Champ", "Ce qu'il signifie", "Ce que vous vérifiez"], [
        ["Action", "BUY, SELL ou WAIT.", "Qu'elle respecte votre plan et le sens autorisé."],
        ["Confiance", "Score de 0 à 100. Le projet utilise un seuil minimal configurable, fixé à 60 % dans la configuration actuelle.", "Qu'elle dépasse votre seuil personnel ; un score élevé ne supprime pas le risque."],
        ["Entrée", "Prix de référence du setup.", "L'écart avec le prix MT5 réel, surtout après un mouvement rapide."],
        ["SL", "Stop Loss, niveau d'invalidation du scénario.", "Qu'il est cohérent avec votre risque et placé au bon côté du marché."],
        ["TP", "Take Profit, objectif de sortie.", "Le ratio rendement/risque. Le minimum configuré est 1,5."],
        ["Lot", "Volume proposé selon les règles de risque et les contraintes du symbole.", "Qu'il reste acceptable pour votre compte et le lot minimal du courtier."],
        ["Justification", "Résumé de la confluence technique et IA.", "Les éléments réellement visibles sur le graphique : tendance, zone, structure, volatilité."]
    ], [1500, 4250, 2750]))
    parts.append(callout("Pourquoi un signal passe en WAIT.", "Le projet force WAIT lorsqu'il n'existe pas de setup institutionnel confirmé, lorsque le prix poursuit déjà excessivement le mouvement, lorsqu'un BUY est en surachat RSI, lorsqu'un SELL est en survente RSI, lorsqu'un trade va contre la tendance haute unité de temps ou lorsqu'il viole le sens autorisé."))

    parts.append(heading("7. Configurer le risque et les marchés", 1))
    parts.append(heading("Réglages essentiels", 2))
    parts.append(table(["Réglage", "Effet pratique", "Conseil de départ"], [
        ["Risque par trade", "Part du capital visée pour le risque du scénario.", "Utilisez une faible valeur en démonstration puis ajustez avec recul."],
        ["Perte max journalière", "Seuil de perte quotidienne servant de garde-fou.", "Définissez une limite que vous acceptez réellement de ne pas dépasser."],
        ["Lot maximal", "Empêche une proposition de volume trop élevé.", "Fixez-le avant tout scan, surtout sur indices et métaux."],
        ["Positions max", "Limite les expositions simultanées en Autopilot.", "Commencez à 1, puis augmentez seulement après validation de votre méthode."],
        ["Objectif quotidien", "Peut arrêter les nouvelles prises de position une fois le seuil atteint.", "Utilisez-le pour éviter de rendre des gains à la volatilité."],
        ["Spread Buffer", "Ajoute une protection lorsque le spread devient trop important par rapport au SL.", "Laissez-le activé, particulièrement aux ouvertures et pendant les annonces."],
        ["Sens de trading", "Autorise BUY et SELL, uniquement BUY, ou uniquement SELL.", "Choisissez le sens qui correspond à votre biais de marché documenté."]
    ], [2000, 4200, 2300]))
    parts.append(heading("Réglage disponible dans l'interface web", 2))
    parts.append(paragraph("Dans le bloc Config, modifiez les champs puis cliquez sur Sauvegarder. La configuration est écrite dans config/config.yaml. La page web permet de modifier directement le risque, la perte maximale, le lot maximal, les positions maximales, le mode, la première unité de temps sélectionnée et les types de marchés autorisés."))
    parts.append(callout("Prudence.", "Changer la configuration modifie les scans suivants et le comportement de l'Autopilot. Désactivez l'Autopilot avant une modification importante, puis vérifiez les nouvelles valeurs affichées.", "FCE4D6"))

    parts.append(heading("8. Choisir le mode d'automatisation", 1))
    parts.append(table(["Mode", "Comportement", "Usage recommandé"], [
        ["assist", "Le bot analyse et affiche les opportunités ; l'utilisateur décide et déclenche l'ordre.", "Mode recommandé pour apprendre et contrôler chaque entrée."],
        ["semi-auto", "Les signaux restent à valider manuellement dans les interfaces du projet.", "À utiliser comme phase de transition après des tests concluants."],
        ["auto", "Avec Autopilot activé, les signaux admissibles peuvent être envoyés à MT5 sans validation manuelle.", "Réservé à une stratégie testée, avec limites de risque conservatrices."]
    ], [1800, 5000, 2700]))
    parts.append(paragraph("Dans l'interface web, l'Autopilot lance un cycle de scan toutes les cinq minutes. Il n'exécute automatiquement que si le mode est auto. Dans les autres modes, il peut scanner et afficher des opportunités, mais l'exécution reste manuelle."))
    parts.append(heading("Garde-fous d'exécution observés", 2))
    for item in [
        "Le signal doit être BUY ou SELL, avec un symbole, un SL, un TP, un lot et un cours de marché valides.",
        "La confiance doit atteindre le seuil minimum configuré et le ratio rendement/risque réel doit atteindre le minimum configuré.",
        "Une politique BUY_ONLY ou SELL_ONLY rejette les signaux dans le sens opposé.",
        "Le bot refuse une nouvelle position si la limite par symbole est atteinte et limite les positions globales pendant l'Autopilot.",
        "L'ordre est soumis à une vérification MT5 avant son exécution. Un refus MT5 n'est pas contourné par l'application."
    ]:
        parts.append(bullet(item))

    parts.append(heading("9. Utiliser le chat et surveiller les positions", 1))
    parts.append(heading("Poser une bonne question au chat", 2))
    parts.append(paragraph("Le chat est un assistant d'explication. Formulez une question précise, avec le symbole, l'unité de temps et l'objectif. Après un scan, ses résultats sont injectés dans le contexte du chat web, ce qui facilite les questions sur la dernière analyse."))
    parts.append(table(["Question imprécise", "Question utile"], [
        ["Le marché monte ?", "Explique le dernier signal sur BTCUSD en M15 : quels éléments justifient BUY, SELL ou WAIT ?"],
        ["Je prends ?", "Quels sont les risques du signal EURUSD affiché : distance au SL, ratio R:R et tendance haute unité de temps ?"],
        ["C'est quoi RSI ?", "Explique simplement pourquoi un RSI élevé peut empêcher un BUY dans le filtre anti-FOMO."]
    ], [4200, 5400]))
    parts.append(heading("Suivre une position", 2))
    parts.append(paragraph("Ouvrez l'onglet ou le bloc Positions ouvertes. Chaque ligne affiche notamment le ticket, le symbole, le type, le volume, le prix d'ouverture, le SL, le TP, le prix actuel et le profit. L'interface web rafraîchit la liste régulièrement et le bouton Positions force un rafraîchissement."))
    parts.append(paragraph("La configuration inclut des mécanismes de protection tels que le break-even, le trailing stop, la protection du profit et des objectifs de gain. La surveillance automatique des positions dépend de l'Autopilot : ne supposez pas qu'une gestion automatique est active lorsqu'il est désactivé."))
    parts.append(heading("Fonctions avancées de la version bureau", 2))
    for item in [
        "BACKTEST MT5 : utilisez-le pour évaluer une logique sur des données historiques avant d'envisager l'Autopilot.",
        "CONSCIENCE ÉVOLUTIVE : consultez les poids adaptatifs, le journal des trades, les auto-critiques et les leçons mémorisées ; utilisez ces informations comme outil de revue, pas comme promesse de performance.",
        "Order Block post-trade : le projet peut effectuer des rescans après une opération ; en mode assist ou semi-auto, les signaux restent à valider, tandis qu'en auto les limites configurées restent applicables."
    ]:
        parts.append(bullet(item))

    parts.append(heading("10. Exemples guidés", 1))
    parts.append(heading("Exemple 1 - Analyser un signal BUY sans se précipiter", 2))
    parts.append(callout("Contexte fictif.", "Vous autorisez Crypto, sélectionnez M15, mode assist et lancez un scan. Une carte affiche : BUY sur BTCUSD, confiance 72 %, entrée 64 000, SL 63 600, TP 64 700, lot 0,01, avec une justification mentionnant une confluence de tendance et une zone technique."))
    for index, step in enumerate([
        "Ouvrez BTCUSD en M15 dans MT5 et vérifiez que le prix de marché n'est pas déjà très éloigné de l'entrée proposée.",
        "Mesurez le risque : l'écart entre entrée et SL est de 400 points dans cet exemple. Vérifiez que le volume proposé est acceptable pour votre compte et votre symbole.",
        "Comparez le TP et le SL : le ratio théorique est 700 / 400 = 1,75, donc supérieur au minimum projet de 1,5.",
        "Relisez la justification et vérifiez qu'elle n'est pas contredite par une annonce ou une structure visible sur MT5.",
        "Sur compte démo, cliquez sur PRENDRE POSITION. Si MT5 refuse l'ordre, lisez le motif et corrigez la cause plutôt que de forcer l'exécution."
    ], start=1):
        parts.append(numbered(index, step))
    parts.append(heading("Exemple 2 - Comprendre WAIT", 2))
    parts.append(callout("Situation fictive.", "Le scanner affiche WAIT (EURUSD) - pas d'opportunité claire. Cela ne signale pas une erreur. Le bot a choisi de préserver le capital plutôt que d'inventer une entrée."))
    parts.append(paragraph("Action recommandée : ne prenez pas de trade à partir de ce résultat. Vous pouvez demander au chat quels filtres ont été déterminants, choisir une unité de temps supérieure ou attendre un prochain scan. N'essayez pas de transformer WAIT en BUY ou SELL par impatience."))
    parts.append(heading("Exemple 3 - Un ordre est refusé", 2))
    parts.append(callout("Situation fictive.", "Un signal SELL paraît intéressant, mais PRENDRE POSITION renvoie un échec. Les causes possibles dans le projet incluent : marché fermé ou tick indisponible, confiance sous le seuil, ratio R:R insuffisant au prix réel, sens de trading incompatible, limite de positions atteinte, objectif journalier atteint ou contrôle MT5 refusé."))
    parts.append(paragraph("Action recommandée : vérifiez d'abord le statut MT5 et la table des positions. Ensuite, contrôlez vos limites et le prix courant. Ne modifiez pas les règles de risque juste pour faire passer un ordre précis."))
    parts.append(heading("Exemple 4 - Autopilot contrôlé", 2))
    for index, step in enumerate([
        "Testez d'abord le même univers de marchés en mode assist pendant plusieurs séances de démonstration.",
        "Fixez une seule position maximale, un lot maximal bas et un profil Prudent.",
        "Passez en auto seulement si vous comprenez les conditions d'entrée et les refus.",
        "Activez AUTOPILOT et surveillez le premier cycle. Sur le web, les scans sont programmés toutes les cinq minutes.",
        "Désactivez immédiatement AUTOPILOT en cas de comportement inattendu, d'annonce majeure ou de perte de connexion MT5."
    ], start=1):
        parts.append(numbered(index, step))

    parts.append(heading("11. Résoudre les problèmes fréquents", 1))
    parts.append(table(["Symptôme", "Cause probable", "Action"], [
        ["MT5 : déconnecté", "Terminal fermé, compte non connecté ou initialisation MT5 impossible.", "Ouvrez MT5 sur le même PC, reconnectez le compte puis relancez Focon Assistant."],
        ["Compte indisponible", "MT5 ne retourne pas les informations de compte.", "Vérifiez la connexion MT5, la session de courtage et les droits du compte."],
        ["Aucun résultat exploitable", "Aucun setup clair ou filtres stricts.", "Acceptez WAIT ; vérifiez les marchés autorisés, l'unité de temps et les symboles visibles."],
        ["Scan déjà en cours", "Un scan s'exécute en arrière-plan.", "Attendez sa fin ou utilisez STOP si vous voulez l'interrompre."],
        ["Ordre refusé", "Règle de risque, prix devenu invalide, limite, marché fermé ou refus MT5.", "Lisez le motif, vérifiez MT5 et vos limites. Ne forcez pas l'ordre."],
        ["Chat en erreur", "Clé API, quota ou fournisseur IA indisponible.", "Vérifiez les clés dans .env et la connexion réseau ; relancez si nécessaire."],
        ["Application ne démarre pas", "Clés Gemini/OpenRouter absentes ou dépendances non installées.", "Complétez .env puis exécutez l'installation des dépendances du projet."]
    ], [2200, 3400, 4000]))
    parts.append(heading("Où chercher plus de détails", 2))
    parts.append(paragraph("Les journaux d'exécution se trouvent dans le dossier logs. Pour un diagnostic technique, conservez le message complet sans y inclure vos clés API. Le script scripts/test_scan.py permet également de tester le scanner sans interface, à utiliser par une personne à l'aise avec le terminal."))

    parts.append(heading("12. Glossaire et checklist", 1))
    parts.append(heading("Glossaire rapide", 2))
    parts.append(table(["Terme", "Définition simplifiée"], [
        ["BUY / SELL", "Ordre d'achat / ordre de vente."],
        ["WAIT", "Aucune entrée jugée suffisamment valide ; ne pas intervenir."],
        ["SL", "Stop Loss : sortie prévue si le scénario est invalidé."],
        ["TP", "Take Profit : objectif de sortie en gain."],
        ["R:R", "Ratio rendement/risque : gain potentiel divisé par perte potentielle."],
        ["EMA", "Moyenne mobile exponentielle, utilisée ici pour lire tendance et extension du prix."],
        ["RSI", "Indicateur de momentum ; le projet l'utilise notamment pour éviter certains achats en surachat et ventes en survente."],
        ["Order Block", "Zone de prix associée à une activité institutionnelle potentielle, recherchée par le scanner."],
        ["AMD", "Accumulation, Manipulation, Distribution : lecture de phases possibles de marché."],
        ["Autopilot", "Boucle qui lance des scans et peut exécuter selon le mode auto et les garde-fous."]
    ], [2600, 7400]))
    parts.append(heading("Checklist avant toute séance", 2))
    checklist = [
        "MT5 est ouvert, connecté et Trading Algorithmique est réglé selon mon intention.",
        "Je travaille en compte de démonstration ou je comprends pleinement le risque d'un ordre réel.",
        "Mon profil de risque, mon lot maximal, ma perte maximale et mes positions maximales sont vérifiés.",
        "Les marchés et le sens de trading autorisés correspondent à mon plan du jour.",
        "Je suis prêt à accepter WAIT et les refus du système sans contourner les garde-fous.",
        "L'Autopilot est désactivé, sauf si je l'ai consciemment préparé et que je peux le surveiller.",
        "Je ne partage ni mon fichier .env ni mes clés API."
    ]
    for item in checklist:
        parts.append(bullet(item))
    parts.append(callout("Rappel final.", "Focon Assistant est le plus utile comme cadre d'analyse et de discipline. La responsabilité du paramétrage, de la validation et du risque reste celle de l'utilisateur.", "FCE4D6"))

    sect = (
        '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>'
        '<w:pgMar w:top="1050" w:right="1050" w:bottom="1050" w:left="1050"/>'
        '<w:footerReference w:type="default" r:id="rId2"/>'
        '</w:sectPr>'
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<w:body>' + ''.join(parts) + sect + '</w:body></w:document>'
    )


def styles_xml():
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Arial" w:hAnsi="Arial" w:cs="Arial"/><w:sz w:val="21"/><w:szCs w:val="21"/></w:rPr></w:rPrDefault></w:docDefaults>
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:qFormat/><w:pPr/><w:rPr><w:rFonts w:ascii="Arial" w:hAnsi="Arial"/><w:sz w:val="21"/><w:color w:val="1F1F1F"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="CoverKicker"><w:name w:val="Cover Kicker"/><w:basedOn w:val="Normal"/><w:rPr><w:b/><w:color w:val="1F4E79"/><w:sz w:val="24"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="Heading 1"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:qFormat/><w:rPr><w:b/><w:color w:val="17365D"/><w:sz w:val="31"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="Heading 2"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:qFormat/><w:rPr><w:b/><w:color w:val="1F4E79"/><w:sz w:val="25"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Code"><w:name w:val="Code"/><w:basedOn w:val="Normal"/><w:rPr><w:rFonts w:ascii="Consolas" w:hAnsi="Consolas"/><w:sz w:val="19"/></w:rPr></w:style>
</w:styles>'''


def footer_xml():
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:ftr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:p><w:pPr><w:jc w:val="center"/></w:pPr><w:r><w:rPr><w:color w:val="6B7C8C"/><w:sz w:val="18"/></w:rPr><w:t>Focon Assistant - Manuel d'utilisation</w:t></w:r></w:p></w:ftr>'''


def write_docx():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    contents = {
        "[Content_Types].xml": '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/><Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/><Override PartName="/word/footer1.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/><Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/><Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/></Types>''',
        "_rels/.rels": '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/><Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/></Relationships>''',
        "word/_rels/document.xml.rels": '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer" Target="footer1.xml"/></Relationships>''',
        "word/document.xml": document_xml(),
        "word/styles.xml": styles_xml(),
        "word/footer1.xml": footer_xml(),
        "docProps/core.xml": f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"><dc:title>Manuel d'utilisation - Focon Assistant</dc:title><dc:subject>Guide utilisateur</dc:subject><dc:creator>Focon Assistant</dc:creator><cp:keywords>trading, MetaTrader 5, assistant, IA</cp:keywords><dcterms:created xsi:type="dcterms:W3CDTF">{date.today().isoformat()}T00:00:00Z</dcterms:created></cp:coreProperties>''',
        "docProps/app.xml": '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"><Application>Focon Assistant</Application><DocSecurity>0</DocSecurity><ScaleCrop>false</ScaleCrop><Company>Focon Assistant</Company><LinksUpToDate>false</LinksUpToDate><SharedDoc>false</SharedDoc><HyperlinksChanged>false</HyperlinksChanged><AppVersion>1.0</AppVersion></Properties>''',
    }
    with ZipFile(OUTPUT, "w", compression=ZIP_DEFLATED) as archive:
        for name, content in contents.items():
            archive.writestr(name, content.encode("utf-8"))
    print(f"Created: {OUTPUT}")


if __name__ == "__main__":
    write_docx()
