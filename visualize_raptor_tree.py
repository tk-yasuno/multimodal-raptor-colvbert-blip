"""
RAPTOR Tree可視化スクリプト
NetworkXとMatplotlibを使用してツリー構造を可視化

保存されたscaling_test_tree_*.pklファイルからRAPTOR Treeを読み込み、
階層構造を視覚化してPNG/PDFとして保存します。
"""

import os
import sys
import pickle
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
from datetime import datetime
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

# 形態素解析
try:
    from fugashi import Tagger, GenericTagger
    FUGASHI_AVAILABLE = True
except ImportError:
    FUGASHI_AVAILABLE = False
    print("⚠️ fugashiがインストールされていません。形態素解析ラベルはTF-IDFのみ使用されます")

# 災害ドメイン語彙
from disaster_vocab import filter_keywords, is_disaster_keyword, DISASTER_DOMAIN_KEYWORDS

# 日本語→英語翻訳辞書（災害用語）
# 日本語→英語翻訳辞書（災害用語）
JA_TO_EN_DICT = {
    # 災害種別
    '災害': 'Disaster', '津波': 'Tsunami', '地震': 'Earthquake', '台風': 'Typhoon', '豪雨': 'Heavy Rain',
    '洪水': 'Flood', '土砂': 'Landslide', '火山': 'Volcano', '噴火': 'Eruption',
    '地滑り': 'Landslide', '崩壊': 'Collapse', '浸水': 'Inundation', '氾濫': 'Overflow',
    '高潮': 'Storm Surge', '竜巻': 'Tornado', '土石流': 'Debris Flow', '大震災': 'Great Earthquake',
    '震災': 'Earthquake Disaster',
    
    # 避難・対応
    '避難': 'Evacuation', '避難所': 'Shelter', '避難場所': 'Evac.Site', '避難者': 'Evacuee',
    '避難指示': 'Evac.Order', '避難勧告': 'Evac.Advisory', '避難経路': 'Evac.Route',
    '避難訓練': 'Drill', '避難行動': 'Evac.Action', '避難計画': 'Evac.Plan', '避難誘導': 'Evac.Guide',
    '救助': 'Rescue', '救援': 'Relief', '救護': 'Aid', '救出': 'Rescue', '捜索': 'Search',
    '救命': 'Lifesaving', '救急': 'Emergency', '応援': 'Support',
    
    # 警報・情報
    '警報': 'Warning', '注意報': 'Advisory', '緊急': 'Emergency', '速報': 'Alert',
    '情報': 'Info', 'アラート': 'Alert', '発令': 'Issue', '解除': 'Lift',
    '伝達': 'Transmission', '通知': 'Notification', '周知': 'Awareness', '広報': 'PR',
    '報告': 'Report', '連絡': 'Contact', '通信': 'Communication',
    
    # 組織・機関
    '自治体': 'Municipality', '市町村': 'Municipality', '都道府県': 'Prefecture',
    '国': 'Nation', '政府': 'Government', '気象庁': 'JMA', '消防': 'Fire Dept',
    '警察': 'Police', '自衛隊': 'SDF', '海上保安庁': 'Coast Guard', '防災': 'Disaster Prev',
    '本部': 'HQ', '対策本部': 'Response HQ', '災害対策': 'Disaster Response',
    '危機管理': 'Crisis Mgmt', '行政': 'Administration', '市': 'City', '町': 'Town',
    '村': 'Village', '県': 'Prefecture', '区': 'Ward',
    
    # 被害・状況
    '被害': 'Damage', '被災': 'Disaster', '被災者': 'Victim', '被災地': 'Affected Area',
    '死者': 'Fatality', '行方不明': 'Missing', '負傷': 'Injury', '倒壊': 'Collapse',
    '損壊': 'Destruction', '冠水': 'Flooding', '孤立': 'Isolation',
    '停電': 'Blackout', '断水': 'Water Outage', '道路': 'Road', '通行止め': 'Road Closed',
    '寸断': 'Cutoff', '全壊': 'Total Collapse', '半壊': 'Partial Collapse',
    '液状': 'Liquefaction',
    '浸水': 'Flooding', '水害': 'Flood Damage', '風害': 'Wind Damage',
    
    # 施設
    '学校': 'School', '病院': 'Hospital', '公民館': 'Community Ctr', '体育館': 'Gym',
    'ホール': 'Hall', '施設': 'Facility', '港': 'Port', '漁港': 'Fishing Port',
    '堤防': 'Levee', 'ダム': 'Dam', '河川': 'River', '海岸': 'Coast',
    '建物': 'Building', '住宅': 'Housing', '住居': 'Residence',
    
    # 住民・地域
    '住民': 'Resident', '市民': 'Citizen', '町民': 'Townspeople', '村民': 'Villager',
    '地域': 'Area', '地区': 'District', '町内': 'Neighborhood', '世帯': 'Household',
    '家族': 'Family', '子供': 'Children', '子ども': 'Children', '児童': 'Children',
    '高齢者': 'Elderly', '要配慮者': 'Vulnerable', '乳幼児': 'Infant', '園児': 'Preschooler',
    '住宅地': 'Residential', '商業地': 'Commercial', '人口': 'Population',
    
    # 対策・計画
    '対策': 'Measure', '計画': 'Plan', 'マニュアル': 'Manual', 'ガイドライン': 'Guideline',
    '訓練': 'Training', '想定': 'Scenario', 'シミュレーション': 'Simulation',
    'ハザードマップ': 'Hazard Map', '防災計画': 'Disaster Plan', '体制': 'System',
    '整備': 'Development', '準備': 'Preparation', '強化': 'Reinforcement',
    '施策': 'Policy', '方針': 'Guideline', '戦略': 'Strategy', '取組': 'Initiative',
    '備え': 'Preparedness',
    
    # 時間・フェーズ
    '発生': 'Occurrence', '直後': 'Immediate', '当日': 'Same Day', '翌日': 'Next Day',
    '数日後': 'Days Later', '復旧': 'Recovery', '復興': 'Reconstruction',
    '早期': 'Early', '迅速': 'Prompt', '即座': 'Instant',
    
    # 固有名詞（地名）
    '日本': 'Japan', '東日本': 'East Japan', '阪神': 'Hanshin', '熊本': 'Kumamoto', '北海道': 'Hokkaido',
    '九州': 'Kyushu', '四国': 'Shikoku', '太平洋': 'Pacific', '日本海': 'Sea of Japan',
    '南海トラフ': 'Nankai Trough', '東海': 'Tokai', '関東': 'Kanto', '東北': 'Tohoku',
    '中部': 'Chubu', '近畿': 'Kinki', '中国': 'Chugoku', '沖縄': 'Okinawa', '関西': 'Kansai',
    '半島': 'Peninsula', '能登半島': 'Noto Peninsula', '能登': 'Noto', '石川': 'Ishikawa', '福島': 'Fukushima',
    '宮城': 'Miyagi', '岩手': 'Iwate', '淡路': 'Awaji', '東峰': 'Toho', '日田': 'Hita',
    '登別': 'Noboribetsu', '札幌': 'Sapporo', '仙台': 'Sendai', '神戸': 'Kobe',
    '広島': 'Hiroshima', '長崎': 'Nagasaki', '鹿児島': 'Kagoshima', '平成': 'Heisei',
    '沿岸': 'Coastal', '市町': 'City-Town',
    
    # 一般的な動詞・形容詞
    '確認': 'Confirm', '実施': 'Implement', '設置': 'Install', '配置': 'Deploy',
    '開設': 'Open', '運営': 'Operate', '提供': 'Provide', '支援': 'Support',
    '必要': 'Necessary', '重要': 'Important', '安全': 'Safety', '危険': 'Danger',
    '適切': 'Appropriate', '有効': 'Effective', '可能': 'Possible',
    
    # その他の重要語
    '明確': 'Clear', '詳細': 'Detail', '具体': 'Specific', '全体': 'Overall',
    '一部': 'Partial', '多数': 'Many', '少数': 'Few', '大規模': 'Large-scale',
    '小規模': 'Small-scale', '中規模': 'Medium-scale', '残念': 'Regrettable',
    '手元': 'On Hand',
    
    # 教訓・経験
    '教訓': 'Lesson', '経験': 'Experience', '事例': 'Case', '記録': 'Record',
    '資料': 'Document', '参考': 'Reference', 'データ': 'Data',
    
    # カタカナ語
    'ページ': 'Page', 'ケース': 'Case', 'レベル': 'Level', 'システム': 'System',
    'センター': 'Center', 'ネットワーク': 'Network', 'エリア': 'Area', 'ケア': 'Care',
    'テキスト': 'Text',
    
    # 追加の一般語彙（白抜き防止）
    '明らか': 'Clear', '確実': 'Certain', '必要': 'Necessary', '重要': 'Important',
    '可能': 'Possible', '困難': 'Difficult', '十分': 'Sufficient', '不足': 'Shortage',
    '問題': 'Problem', '課題': 'Issue', '改善': 'Improvement', '変更': 'Change',
    '増加': 'Increase', '減少': 'Decrease', '維持': 'Maintain', '強化': 'Strengthen',
    '整備': 'Preparation', '設置': 'Installation', '配置': 'Placement', '移動': 'Movement',
    '開始': 'Start', '終了': 'End', '継続': 'Continue', '中止': 'Suspend',
    '判断': 'Judgment', '決定': 'Decision', '選択': 'Choice', '検討': 'Consideration',
    '調査': 'Survey', '分析': 'Analysis', '評価': 'Evaluation', '確認': 'Confirmation',
    '連絡': 'Contact', '連携': 'Cooperation', '協力': 'Collaboration', '支援': 'Support',
    '管理': 'Management', '運営': 'Operation', '活動': 'Activity', '行動': 'Action',
    '指導': 'Guidance', '教育': 'Education', '学習': 'Learning', '理解': 'Understanding',
    '説明': 'Explanation', '指示': 'Instruction', '要請': 'Request', '依頼': 'Request',
    '報道': 'Report', '公表': 'Announcement', '発表': 'Publication', '通報': 'Notification',
    '受信': 'Reception', '送信': 'Transmission', '配信': 'Distribution', '提供': 'Provision',
    '利用': 'Use', '使用': 'Usage', '活用': 'Utilization', '適用': 'Application',
    '実行': 'Execution', '遂行': 'Performance', '達成': 'Achievement', '完了': 'Completion',
    # エラーログから追加（頻出未登録語）
    '心身': 'Mind-Body', '複数': 'Multiple', '対応': 'Response', '把握': 'Grasp',
    '東北大': 'Tohoku Univ', '文化財': 'Cultural Property', '契機': 'Opportunity',
    '仮設': 'Temporary', '中心': 'Center', '各地': 'Various Places', '北部': 'Northern Part',
    '混乱': 'Confusion', '確保': 'Secure', '番号': 'Number', '遅れ': 'Delay',
    '原発': 'Nuclear Plant', '混雑': 'Congestion', '数値': 'Numerical Value',
}

# pykakasiのグローバル初期化（1回のみ）
_KAKASI_INSTANCE = None

def get_kakasi():
    """pykakasiインスタンスを取得（シングルトン）"""
    global _KAKASI_INSTANCE
    if _KAKASI_INSTANCE is None:
        try:
            import pykakasi
            _KAKASI_INSTANCE = pykakasi.kakasi()
        except ImportError:
            _KAKASI_INSTANCE = False  # インポート失敗をマーク
    return _KAKASI_INSTANCE

def translate_keyword(keyword: str) -> str:
    """
    日本語キーワードを英語に翻訳（辞書ベース + ローマ字化）
    
    1. 既にASCIIならそのまま返す
    2. 翻訳辞書から検索
    3. pykakasiでローマ字化
    4. 失敗時はフォールバック
    """
    # 既に英数字のみの場合はそのまま返す
    if keyword.isascii():
        return keyword
    
    # 辞書から翻訳を取得
    translated = JA_TO_EN_DICT.get(keyword)
    if translated:
        return translated
    
    # pykakasiでローマ字化を試行
    kks = get_kakasi()
    if kks and kks is not False:
        try:
            result = kks.convert(keyword)
            # hepburnキーでローマ字を取得し、単語ごとに先頭大文字化
            romaji_parts = []
            for item in result:
                if 'hepburn' in item and item['hepburn']:
                    romaji_parts.append(item['hepburn'].capitalize())
            
            if romaji_parts:
                return ''.join(romaji_parts)
                
        except Exception as e:
            # デバッグ: エラー内容を出力
            print(f"      ⚠️ ローマ字化エラー '{keyword}': {e}")
    
    # フォールバック: ASCIIのみの代替表現を返す
    import re
    # 数字が含まれている場合は数字を抽出
    numbers = re.findall(r'\d+', keyword)
    if numbers:
        return f"Item{numbers[0]}"
    # それ以外は汎用的な表現（デバッグ出力）
    print(f"      ⚠️ 翻訳失敗 '{keyword}' → 'Term'")
    return "Term"

# UTF-8出力設定（Windows対応）
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 日本語フォント設定
matplotlib.rcParams['font.family'] = ['MS Gothic', 'Yu Gothic', 'Meiryo', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# プロジェクトルート追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '0_base_tsunami-lesson-rag'))


class RAPTORTreeVisualizer:
    """RAPTOR Treeの可視化クラス"""
    
    def __init__(self, tree_path: str, use_morphology: bool = True, top_n: int = 2):
        """
        Args:
            tree_path: RAPTORツリーのpickleファイルパス
            use_morphology: 形態素解析を使用するか（True: MeCab形態素解析、False: TF-IDFのみ）
            top_n: ノードラベルに表示するキーワード数（デフォルト: 2）
        """
        self.tree_path = Path(tree_path)
        self.tree = None
        self.graph = None
        self.node_depths = {}
        self.node_types = {}  # 'leaf' or 'internal'
        self.node_keywords = {}  # ノードごとのキーワード
        self.all_summaries = []  # TF-IDF計算用の全サマリーテキスト
        self.node_to_summary_idx = {}  # ノードIDからサマリーインデックスへのマッピング
        self.node_parents = {}  # ノードIDから親ノードIDへのマッピング（階層的除外用）
        self.used_keywords_by_depth = {}  # 深度ごとに使用済みキーワードを記録
        self.top_n = top_n  # ノードラベルのキーワード数
        
        # 形態素解析設定
        self.use_morphology = use_morphology and FUGASHI_AVAILABLE
        if self.use_morphology:
            try:
                import os
                
                # MeCabの設定ファイルパスを環境変数に設定
                mecabrc_paths = [
                    r'C:\Program Files\MeCab\etc\mecabrc',
                    r'C:\Program Files (x86)\MeCab\etc\mecabrc',
                    r'C:\mecab\etc\mecabrc'
                ]
                
                for mecabrc_path in mecabrc_paths:
                    if os.path.exists(mecabrc_path):
                        os.environ['MECABRC'] = mecabrc_path
                        print(f"   ⚙️ MECABRC: {mecabrc_path}")
                        break
                
                # MeCabの辞書パス設定
                # 優先順位: 1) unidic-lite, 2) システムMeCab, 3) デフォルト
                dicdir = None
                
                # 1. unidic-liteを試行
                try:
                    import unidic_lite
                    dicdir = unidic_lite.DICDIR
                    print(f"   📖 辞書: unidic-lite ({dicdir})")
                except ImportError:
                    pass
                
                # 2. システムMeCabを試行（C:\Program Files\MeCab）
                if dicdir is None:
                    mecab_dic_paths = [
                        r'C:\Program Files\MeCab\dic\ipadic',
                        r'C:\Program Files (x86)\MeCab\dic\ipadic',
                        r'C:\mecab\dic\ipadic'
                    ]
                    for path in mecab_dic_paths:
                        if os.path.exists(path):
                            dicdir = path
                            print(f"   📖 辞書: システムMeCab ({dicdir})")
                            break
                
                # Tagger初期化（GenericTagger for IPADIC）
                if dicdir:
                    # IPADIC形式の辞書にはGenericTaggerを使用
                    if 'ipadic' in dicdir.lower():
                        self.tagger = GenericTagger(f'-d "{dicdir}"')
                        print("   🔧 Tagger: GenericTagger (IPADIC対応)")
                    else:
                        # UniDic形式
                        self.tagger = Tagger(f'-d "{dicdir}"')
                        print("   🔧 Tagger: Tagger (UniDic)")
                else:
                    # 辞書パス指定なし（デフォルト）
                    try:
                        self.tagger = Tagger()
                        print("   📖 辞書: デフォルト (Tagger)")
                    except:
                        self.tagger = GenericTagger()
                        print("   📖 辞書: デフォルト (GenericTagger)")
                
                print("✅ 形態素解析モード: MeCab (fugashi) + 災害ドメイン語彙フィルタ")
            except Exception as e:
                print(f"⚠️ 形態素解析初期化失敗: {e}")
                print("   TF-IDFモードにフォールバック")
                self.use_morphology = False
                self.tagger = None
        else:
            self.tagger = None
            if use_morphology and not FUGASHI_AVAILABLE:
                print("⚠️ fugashi未インストール。TF-IDFモードで動作します")
            else:
                print("📊 TF-IDFモード")
    
    def extract_keywords_morphology(self, text: str, top_n: int = 10) -> List[str]:
        """
        形態素解析を使用してキーワード抽出（災害ドメイン特化）
        
        Args:
            text: 解析するテキスト
            top_n: 抽出するキーワード数（候補を多めに取得）
        
        Returns:
            キーワードリスト（体言止め、ドメイン語彙優先）
        """
        if not self.use_morphology or not text:
            return []
        
        # 形態素解析実行
        words = []
        for word in self.tagger(text):
            # 品詞取得（GenericTaggerとTaggerの両対応）
            if hasattr(word, 'feature'):
                # Tagger (UniDic)
                pos = word.feature.pos1 if hasattr(word.feature, 'pos1') else word.feature[0]
            else:
                # GenericTagger (IPADIC) - featuresは文字列のタプル
                features = word.features if hasattr(word, 'features') else str(word).split('\t')[1].split(',')
                pos = features[0] if features else ''
            
            # 名詞・固有名詞のみ抽出
            if pos in ['名詞', '固有名詞']:
                surface = word.surface if hasattr(word, 'surface') else str(word).split('\t')[0]
                # 1文字は除外（助詞的な名詞を排除）
                if len(surface) >= 2:
                    words.append(surface)
        
        # 災害ドメイン語彙でフィルタリング＋優先順位付け
        # 戦略: 上位候補を多めに取得して、後で階層的除外を適用
        from disaster_vocab import DISASTER_DOMAIN_KEYWORDS
        
        # 頻出する超一般的な災害語彙（各ノード共通になりやすい）
        common_disaster_words = {'津波', '避難', '地震', '被害', '住民', '対策', '発生', '情報'}
        
        filtered = filter_keywords(words, prioritize_domain=True)
        
        # 戦略: ドメイン語彙の中でも共通語を避け、特徴的な語を優先
        specific_keywords = []
        common_keywords = []
        
        # 【改善】候補数を50語に拡大（階層的除外により多くが削除されるため）
        for word in filtered[:50]:
            if word in common_disaster_words:
                common_keywords.append(word)
            else:
                specific_keywords.append(word)
        
        # 特徴的な語を優先、足りなければ共通語で補完
        result = specific_keywords + common_keywords
        
        return result[:top_n]
    
    def extract_keywords_tfidf(self, texts: List[str], top_n: int = 10) -> List[List[str]]:
        """
        TF-IDFを使用して各テキストから重要なキーワードを抽出
        災害ドメイン語彙を優先し、ストップワードを除外
        共通語を避けて特徴的な語を選択
        
        Args:
            texts: サマリーテキストのリスト
            top_n: 各テキストから抽出するキーワード数（候補を多めに取得）
        
        Returns:
            各テキストのキーワードリスト
        """
        if not texts or len(texts) == 0:
            return []
        
        # TF-IDFベクトライザー設定
        # token_pattern: 日本語（2文字以上）と英語（3文字以上）をマッチ
        vectorizer = TfidfVectorizer(
            token_pattern=r'[一-龥ぁ-んァ-ヶー]{2,}|[a-zA-Z]{3,}',
            max_features=1000,
            stop_words=None
        )
        
        # 災害ドメイン語彙をインポート
        from disaster_vocab import STOP_WORDS, is_disaster_keyword
        
        # 頻出する超一般的な災害語彙（各ノード共通になりやすい）
        common_disaster_words = {'津波', '避難', '地震', '被害', '住民', '対策', '発生', '情報'}
        
        try:
            # TF-IDF行列を計算
            tfidf_matrix = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # 各テキストのキーワード抽出
            all_keywords = []
            for i in range(len(texts)):
                # i番目のテキストのTF-IDFスコアを取得
                tfidf_scores = tfidf_matrix[i].toarray()[0]
                
                # スコアでソート（降順）
                top_indices = tfidf_scores.argsort()[::-1]
                
                # 災害ドメイン語彙と一般語を分類
                specific_keywords = []  # 特徴的な語
                common_keywords = []    # 共通の災害語
                other_keywords = []     # その他
                
                # 【改善】候補数を50語に拡大（階層的除外により多くが削除されるため）
                for idx in top_indices[:50]:
                    word = feature_names[idx]
                    if tfidf_scores[idx] <= 0:
                        continue
                    
                    # ストップワード除外
                    if word in STOP_WORDS:
                        continue
                    
                    # 分類
                    if word in common_disaster_words:
                        common_keywords.append(word)
                    elif is_disaster_keyword(word):
                        specific_keywords.append(word)
                    else:
                        other_keywords.append(word)
                
                # 優先順位: 特徴的災害語 > その他 > 共通災害語
                keywords = (specific_keywords + other_keywords + common_keywords)[:top_n]
                all_keywords.append(keywords)
            
            return all_keywords
            
        except Exception as e:
            print(f"⚠️ TF-IDF抽出エラー: {e}")
            # エラー時は空リストを返す
            return [[] for _ in texts]
        
    def load_tree(self):
        """Pickleファイルからツリーを読み込み"""
        print(f"📂 ツリー読み込み中: {self.tree_path.name}")
        
        with open(self.tree_path, 'rb') as f:
            data = pickle.load(f)
        
        # 辞書形式の場合、'tree'キーからツリーシステムを取得
        if isinstance(data, dict) and 'tree' in data:
            self.tree = data['tree']
            print(f"✅ ツリー読み込み完了（辞書形式）")
        else:
            self.tree = data
            print(f"✅ ツリー読み込み完了")
        
        return self.tree
    
    def build_graph(self):
        """NetworkXグラフを構築"""
        print("🔨 NetworkXグラフ構築中...")
        
        self.graph = nx.DiGraph()
        
        # ノードカウンター
        self.node_counter = 0
        self.leaf_nodes = []
        self.internal_nodes = []
        
        # 第1パス: 全サマリーテキストを収集
        temp_nodes = []  # (node_id, cluster_id, summary_text, depth, parent_id, has_children) のリスト
        
        def get_node_id():
            """ユニークなノードIDを生成"""
            self.node_counter += 1
            return f"node_{self.node_counter}"
        
        def collect_summaries(tree_dict, depth=0, parent_id=None):
            """
            第1パス: サマリーテキストを収集
            """
            if not isinstance(tree_dict, dict):
                return
            
            if 'clusters' in tree_dict and tree_dict['clusters']:
                clusters = tree_dict['clusters']
                
                for cluster_id, cluster_data in clusters.items():
                    # サマリーテキスト抽出
                    summary_text = ""
                    if 'summary' in cluster_data:
                        summary = cluster_data['summary']
                        if hasattr(summary, 'page_content'):
                            summary_text = summary.page_content
                        elif isinstance(summary, dict):
                            summary_text = summary.get('page_content', summary.get('text', ''))
                    
                    # 子の有無判定
                    has_children = ('children' in cluster_data and 
                                  cluster_data['children'] and 
                                  'clusters' in cluster_data['children'] and
                                  cluster_data['children']['clusters'])
                    
                    # ノードID生成
                    node_id = get_node_id()
                    
                    # 情報を保存
                    temp_nodes.append({
                        'node_id': node_id,
                        'cluster_id': cluster_id,
                        'summary_text': summary_text,
                        'depth': depth,
                        'parent_id': parent_id,
                        'has_children': has_children
                    })
                    
                    # サマリーテキストをリストに追加
                    self.all_summaries.append(summary_text if summary_text else "")
                    self.node_to_summary_idx[node_id] = len(self.all_summaries) - 1
                    
                    # 子ツリーを再帰的に処理
                    if has_children:
                        collect_summaries(
                            cluster_data['children'],
                            depth + 1,
                            node_id
                        )
        
        # 第1パス実行: サマリー収集
        if isinstance(self.tree, dict):
            collect_summaries(self.tree)
        else:
            print("❌ エラー: ツリーが辞書形式ではありません")
            return None
        
        print(f"   第1パス完了: {len(self.all_summaries)}個のサマリーを収集")
        
        # 親子関係マップ構築
        for node_info in temp_nodes:
            node_id = node_info['node_id']
            parent_id = node_info['parent_id']
            if parent_id:
                self.node_parents[node_id] = parent_id
        
        # 第2パス: キーワード抽出（形態素解析 or TF-IDF）
        if self.use_morphology:
            # 形態素解析モード
            print("   🧠 形態素解析でキーワード抽出中（災害ドメイン特化 + 階層的除外）...")
            all_keywords = []
            for summary_text in self.all_summaries:
                # 【改善】上位20語の候補を取得（後で階層的除外を適用）
                keywords = self.extract_keywords_morphology(summary_text, top_n=20)
                all_keywords.append(keywords)
        else:
            # TF-IDFモード（従来）
            print("   📊 TF-IDFキーワード抽出中（階層的除外）...")
            if len(self.all_summaries) > 0:
                # 【改善】上位20語の候補を取得（後で階層的除外を適用）
                all_keywords = self.extract_keywords_tfidf(self.all_summaries, top_n=20)
            else:
                all_keywords = []
        
        # 第3パス: グラフ構築（深度順にソートして処理）
        print("   グラフノード作成中（階層的キーワード除外適用）...")
        
        # 深度順にソート（上位から下位へ）
        temp_nodes_sorted = sorted(temp_nodes, key=lambda x: x['depth'])
        
        for node_info in temp_nodes_sorted:
            node_id = node_info['node_id']
            cluster_id = node_info['cluster_id']
            summary_text = node_info['summary_text']
            depth = node_info['depth']
            parent_id = node_info['parent_id']
            has_children = node_info['has_children']
            
            # キーワード取得
            summary_idx = self.node_to_summary_idx[node_id]
            raw_keywords = all_keywords[summary_idx] if summary_idx < len(all_keywords) else []
            
            # 【改善】階層的除外: より厳密な祖先キーワード収集
            ancestor_keywords = set()
            
            # 1. 親の系譜を辿って、全ての祖先ノードのキーワードを収集
            current_parent = parent_id
            while current_parent:
                if current_parent in self.node_keywords:
                    ancestor_keywords.update(self.node_keywords[current_parent])
                current_parent = self.node_parents.get(current_parent)
            
            # 2. より浅い全ての深度で使用されたキーワードも除外
            # 例: depth=2のノードは、depth=0とdepth=1の全キーワードを除外
            for d in range(depth):
                if d in self.used_keywords_by_depth:
                    ancestor_keywords.update(self.used_keywords_by_depth[d])
            
            # 3. 現在の深度で既に使用されたキーワードも除外（兄弟ノード）
            if depth in self.used_keywords_by_depth:
                ancestor_keywords.update(self.used_keywords_by_depth[depth])
            
            # 【改善】祖先と重複しないキーワードのみ選択
            # まず、完全に除外されるべきキーワードをフィルタリング
            filtered_keywords = []
            remaining_keywords = []
            
            for kw in raw_keywords:
                if kw not in ancestor_keywords:
                    remaining_keywords.append(kw)
            
            # 【改善2】同一ノード内でのキーワード重複を防ぐ
            # remaining_keywordsから重複なしでtop_n個選択
            seen_in_node = set()
            for kw in remaining_keywords:
                if kw not in seen_in_node:
                    filtered_keywords.append(kw)
                    seen_in_node.add(kw)
                    if len(filtered_keywords) >= self.top_n:
                        break
            
            # 候補が不足している場合の警告
            if len(filtered_keywords) < self.top_n:
                print(f"      ⚠️ ノード{node_id} (depth={depth}): キーワード不足 ({len(filtered_keywords)}/{self.top_n})")
            
            keywords = filtered_keywords[:self.top_n] if filtered_keywords else []
            self.node_keywords[node_id] = keywords
            
            # 使用済みキーワードを記録
            if depth not in self.used_keywords_by_depth:
                self.used_keywords_by_depth[depth] = set()
            self.used_keywords_by_depth[depth].update(keywords)
            
            # ノードタイプ判定
            node_type = 'internal' if has_children else 'leaf'
            
            if node_type == 'leaf':
                self.leaf_nodes.append(node_id)
            else:
                self.internal_nodes.append(node_id)
            
            self.node_types[node_id] = node_type
            self.node_depths[node_id] = depth
            
            # キーワードラベル作成（体言止め、簡潔に）
            if keywords:
                if self.use_morphology:
                    # 形態素解析: 体言止めを「・」で接続
                    # 例: "避難・警報・訓練" (top_n=3の場合)
                    keyword_label = "・".join(keywords[:self.top_n])
                else:
                    # TF-IDF: 体言止めを「・」で接続（災害ドメイン優先）
                    keyword_label = "・".join(keywords[:self.top_n])
            else:
                keyword_label = f"C{cluster_id}"
            
            self.graph.add_node(
                node_id,
                depth=depth,
                text=str(summary_text[:100]) if summary_text else "",
                keywords=keywords,
                node_type=node_type,
                label=keyword_label
            )
            
            # 親ノードとの接続
            if parent_id:
                self.graph.add_edge(parent_id, node_id)
        
        if self.graph.number_of_nodes() == 0:
            print("❌ エラー: ノードが抽出できませんでした")
            return None
        
        print(f"✅ グラフ構築完了")
        print(f"   総ノード数: {self.graph.number_of_nodes()}")
        print(f"   リーフノード: {len(self.leaf_nodes)}")
        print(f"   内部ノード: {len(self.internal_nodes)}")
        print(f"   エッジ数: {self.graph.number_of_edges()}")
        print(f"   最大深度: {max(self.node_depths.values()) if self.node_depths else 0}")
        
        return self.graph
    
    def visualize_tree(self, output_path: str, figsize: Tuple[int, int] = None, dpi: int = 150):
        """
        ツリーを可視化して保存
        
        Args:
            output_path: 出力ファイルパス（PNG/PDF）
            figsize: 図のサイズ（幅、高さ）。Noneの場合はノード数に応じて自動調整
            dpi: 解像度
        """
        if self.graph is None:
            print("❌ エラー: グラフが構築されていません。先にbuild_graph()を実行してください")
            return
        
        print(f"🎨 ツリー可視化中...")
        
        # ノード数に応じて図のサイズを自動調整
        if figsize is None:
            total_nodes = self.graph.number_of_nodes()
            if total_nodes > 30:
                figsize = (28, 14)  # 大きなツリー
            elif total_nodes > 20:
                figsize = (24, 12)  # 中規模ツリー
            else:
                figsize = (20, 10)  # 小規模ツリー
        
        # 図とサブプロット作成
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # 階層レイアウト計算
        pos = self._compute_hierarchical_layout()
        
        # ノードの色設定
        node_colors = []
        for node in self.graph.nodes():
            if self.node_types[node] == 'leaf':
                node_colors.append('#90EE90')  # ライトグリーン（リーフ）
            else:
                depth = self.node_depths[node]
                # 深さに応じて色を変化（青系のグラデーション）
                intensity = 1.0 - (depth * 0.3)
                node_colors.append((0.3, 0.5, intensity))
        
        # ノードサイズ設定（よりコンパクトに）
        node_sizes = []
        for node in self.graph.nodes():
            if self.node_types[node] == 'leaf':
                node_sizes.append(500)  # リーフノードを少し大きく
            else:
                depth = self.node_depths[node]
                size = 1000 + (depth * 300)  # 内部ノードも調整
                node_sizes.append(size)
        
        # エッジ描画
        nx.draw_networkx_edges(
            self.graph, pos, ax=ax,
            edge_color='gray',
            alpha=0.5,
            arrows=True,
            arrowsize=15,
            width=1.5,
            connectionstyle='arc3,rad=0.1'
        )
        
        # ノード描画
        nx.draw_networkx_nodes(
            self.graph, pos, ax=ax,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.9,
            edgecolors='black',
            linewidths=2
        )
        
        # ラベル描画（キーワード表示）
        labels = nx.get_node_attributes(self.graph, 'label')
        
        # ラベル位置をノードの少し下に調整
        label_pos = {}
        y_offset = 0.08  # Y軸方向のオフセット（ノードの下に配置）
        for node, (x, y) in pos.items():
            label_pos[node] = (x, y - y_offset)
        
        # フォントサイズを調整（ノード数に応じて）
        total_nodes = self.graph.number_of_nodes()
        if total_nodes > 50:
            font_size = 5
        elif total_nodes > 30:
            font_size = 6
        elif total_nodes > 20:
            font_size = 7
        else:
            font_size = 8
        
        nx.draw_networkx_labels(
            self.graph, label_pos, labels, ax=ax,
            font_size=font_size,
            font_weight='bold',
            font_family='MS Gothic',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='gray', alpha=0.85)
        )
        
        # タイトル設定
        tree_name = self.tree_path.stem
        total_nodes = self.graph.number_of_nodes()
        max_depth = max(self.node_depths.values()) + 1
        leaf_count = sum(1 for t in self.node_types.values() if t == 'leaf')
        internal_count = total_nodes - leaf_count
        
        ax.set_title(
            f"RAPTOR Tree: {tree_name}\n"
            f"Total Nodes: {total_nodes} (Leaf: {leaf_count}, Internal: {internal_count}) | "
            f"Max Depth: {max_depth}",
            fontsize=14,
            fontweight='bold',
            pad=20
        )
        
        # 凡例追加
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#90EE90', edgecolor='black', label='リーフノード (Depth 0)'),
            Patch(facecolor=(0.3, 0.5, 0.7), edgecolor='black', label='内部ノード (Depth 1+)')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        ax.axis('off')
        plt.tight_layout()
        
        # 保存
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ 可視化完了: {output_path}")
        print(f"   ファイルサイズ: {output_path.stat().st_size / 1024:.1f} KB")
        
        # 【追加】英語版も作成
        self.visualize_tree_en(output_path, figsize, dpi)
    
    def visualize_tree_en(self, ja_output_path: str, figsize: Tuple[int, int] = None, dpi: int = 150):
        """
        ツリーを英語で可視化して保存（日本語版の英訳）
        
        Args:
            ja_output_path: 日本語版の出力ファイルパス
            figsize: 図のサイズ
            dpi: 解像度
        """
        # 英語版のファイル名を生成（_ENを追加）
        ja_path = Path(ja_output_path)
        en_filename = ja_path.stem + '_EN' + ja_path.suffix
        en_output_path = ja_path.parent / en_filename
        
        print(f"🌍 英語版ツリー可視化中...")
        
        # ノード数に応じて図のサイズを自動調整
        if figsize is None:
            total_nodes = self.graph.number_of_nodes()
            if total_nodes > 30:
                figsize = (28, 14)
            elif total_nodes > 20:
                figsize = (24, 12)
            else:
                figsize = (20, 10)
        
        # 図とサブプロット作成
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # 階層レイアウト計算（日本語版と同じ）
        pos = self._compute_hierarchical_layout()
        
        # ノードの色設定（日本語版と同じ）
        node_colors = []
        for node in self.graph.nodes():
            if self.node_types[node] == 'leaf':
                node_colors.append('#90EE90')
            else:
                depth = self.node_depths[node]
                intensity = 1.0 - (depth * 0.3)
                node_colors.append((0.3, 0.5, intensity))
        
        # ノードサイズ設定（日本語版と同じ）
        node_sizes = []
        for node in self.graph.nodes():
            if self.node_types[node] == 'leaf':
                node_sizes.append(500)
            else:
                depth = self.node_depths[node]
                size = 1000 + (depth * 300)
                node_sizes.append(size)
        
        # エッジ描画
        nx.draw_networkx_edges(
            self.graph, pos, ax=ax,
            edge_color='gray',
            alpha=0.5,
            arrows=True,
            arrowsize=15,
            width=1.5,
            connectionstyle='arc3,rad=0.1'
        )
        
        # ノード描画
        nx.draw_networkx_nodes(
            self.graph, pos, ax=ax,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.9,
            edgecolors='black',
            linewidths=2
        )
        
        # 【重要】ラベルを英語に翻訳
        ja_labels = nx.get_node_attributes(self.graph, 'label')
        en_labels = {}
        for node_id, ja_label in ja_labels.items():
            # 「・」で分割して各キーワードを翻訳
            keywords = ja_label.split('・')
            en_keywords = [translate_keyword(kw) for kw in keywords]
            en_labels[node_id] = ' · '.join(en_keywords)  # 英語では中黒の代わりにスペース+·+スペース
        
        # ラベル位置をノードの少し下に調整
        label_pos = {}
        y_offset = 0.08
        for node, (x, y) in pos.items():
            label_pos[node] = (x, y - y_offset)
        
        # フォントサイズを調整（ノード数に応じて）
        total_nodes = self.graph.number_of_nodes()
        if total_nodes > 50:
            font_size = 5
        elif total_nodes > 30:
            font_size = 6
        elif total_nodes > 20:
            font_size = 7
        else:
            font_size = 8
        
        nx.draw_networkx_labels(
            self.graph, label_pos, en_labels, ax=ax,
            font_size=font_size,
            font_weight='bold',
            font_family='DejaVu Sans',  # 英語フォント
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='gray', alpha=0.85)
        )
        
        # タイトル設定（英語）
        tree_name = self.tree_path.stem
        total_nodes = self.graph.number_of_nodes()
        max_depth = max(self.node_depths.values()) + 1
        leaf_count = sum(1 for t in self.node_types.values() if t == 'leaf')
        internal_count = total_nodes - leaf_count
        
        ax.set_title(
            f"RAPTOR Tree: {tree_name}\n"
            f"Total Nodes: {total_nodes} (Leaf: {leaf_count}, Internal: {internal_count}) | "
            f"Max Depth: {max_depth}",
            fontsize=14,
            fontweight='bold',
            pad=20
        )
        
        # 凡例追加（英語）
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#90EE90', edgecolor='black', label='Leaf Node (Depth 0)'),
            Patch(facecolor=(0.3, 0.5, 0.7), edgecolor='black', label='Internal Node (Depth 1+)')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        ax.axis('off')
        plt.tight_layout()
        
        # 保存
        en_output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(en_output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ 英語版可視化完了: {en_output_path}")
        print(f"   ファイルサイズ: {en_output_path.stat().st_size / 1024:.1f} KB")
    
    def _compute_hierarchical_layout(self) -> Dict[str, Tuple[float, float]]:
        """階層レイアウトを計算（ラベル重複を防ぐ）"""
        pos = {}
        
        # 深度ごとにノードをグループ化
        depth_groups = {}
        for node, depth in self.node_depths.items():
            if depth not in depth_groups:
                depth_groups[depth] = []
            depth_groups[depth].append(node)
        
        max_depth = max(self.node_depths.values())
        
        # 各深度で必要な最小幅を計算
        # リーフノードが多い場合は幅を広げる
        max_nodes_in_depth = max(len(nodes) for nodes in depth_groups.values())
        
        # ラベルの重複を避けるための最小間隔を設定
        # ノード数が多いほど広い範囲に配置
        if max_nodes_in_depth > 15:
            x_margin = 0.02
            x_range = (0.01, 0.99)
        elif max_nodes_in_depth > 10:
            x_margin = 0.05
            x_range = (0.03, 0.97)
        else:
            x_margin = 0.1
            x_range = (0.05, 0.95)
        
        # 各深度のノードを配置
        for depth, nodes in depth_groups.items():
            y = max_depth - depth  # 上から下へ（ルートが上）
            n = len(nodes)
            
            # X座標を均等配置（広い範囲で）
            if n == 1:
                x_coords = [0.5]
            else:
                x_coords = np.linspace(x_range[0], x_range[1], n)
            
            for i, node in enumerate(nodes):
                pos[node] = (x_coords[i], y)
        
        return pos
    
    def create_statistics_plot(self, output_path: str):
        """ツリー統計のプロット作成"""
        print("📊 統計プロット作成中...")
        
        # 深度ごとのノード数集計
        depth_counts = {}
        for node, depth in self.node_depths.items():
            depth_counts[depth] = depth_counts.get(depth, 0) + 1
        
        depths = sorted(depth_counts.keys())
        counts = [depth_counts[d] for d in depths]
        
        # プロット
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=100)
        
        # 1. 深度別ノード数
        colors = plt.cm.viridis(np.linspace(0, 1, len(depths)))
        ax1.bar(depths, counts, color=colors, edgecolor='black', linewidth=1.5)
        ax1.set_xlabel('深度 (Depth)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('ノード数', fontsize=12, fontweight='bold')
        ax1.set_title('深度別ノード数', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # 値を棒グラフ上に表示
        for i, (d, c) in enumerate(zip(depths, counts)):
            ax1.text(d, c + max(counts)*0.02, str(c), ha='center', va='bottom', fontweight='bold')
        
        # 2. ツリー統計サマリー
        total_nodes = sum(counts)
        leaf_nodes = counts[0] if 0 in depth_counts else 0
        internal_nodes = total_nodes - leaf_nodes
        max_depth = max(depths)
        avg_branching = internal_nodes / max(1, max_depth)
        
        stats_text = (
            f"総ノード数: {total_nodes}\n"
            f"リーフノード: {leaf_nodes}\n"
            f"内部ノード: {internal_nodes}\n"
            f"最大深度: {max_depth}\n"
            f"平均分岐数: {avg_branching:.2f}\n"
            f"リーフ比率: {leaf_nodes/total_nodes*100:.1f}%"
        )
        
        ax2.text(0.5, 0.5, stats_text, 
                 ha='center', va='center',
                 fontsize=14,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                 family='MS Gothic')
        ax2.axis('off')
        ax2.set_title('ツリー統計サマリー', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # 保存
        output_path = Path(output_path)
        plt.savefig(output_path, dpi=100, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ 統計プロット保存: {output_path}")


def visualize_all_trees(
    pkl_dir: str,
    output_dir: str,
    pattern: str = "scaling_test_tree_*.pkl",
    use_morphology: bool = True,
    top_n: int = 2
):
    """
    指定ディレクトリ内の全RAPTORツリーを可視化
    
    Args:
        pkl_dir: pklファイルがあるディレクトリ
        output_dir: 出力先ディレクトリ
        pattern: ファイル名パターン
        use_morphology: 形態素解析を使用するか
        top_n: ノードラベルに表示するキーワード数（デフォルト: 2）
    """
    pkl_dir = Path(pkl_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # pklファイルを検索
    pkl_files = list(pkl_dir.glob(pattern))
    
    if not pkl_files:
        print(f"❌ エラー: {pkl_dir} に {pattern} が見つかりません")
        return
    
    print("=" * 80)
    print("RAPTOR Tree 可視化")
    print(f"検出ファイル数: {len(pkl_files)}")
    print(f"🧠 モード: {'形態素解析 (MeCab + 災害語彙)' if use_morphology else 'TF-IDF + 災害ドメイン語彙フィルタ'}")
    print(f"🏷️  キーワード数: {top_n}個/ノード")
    print("=" * 80)
    
    for i, pkl_file in enumerate(sorted(pkl_files), 1):
        print(f"\n[{i}/{len(pkl_files)}] 処理中: {pkl_file.name}")
        print("-" * 80)
        
        try:
            # 可視化インスタンス作成（形態素解析モード指定）
            visualizer = RAPTORTreeVisualizer(pkl_file, use_morphology=use_morphology, top_n=top_n)
            
            # ツリー読み込み
            visualizer.load_tree()
            
            # グラフ構築
            visualizer.build_graph()
            
            # 出力ファイル名生成
            base_name = pkl_file.stem
            tree_png = output_dir / f"{base_name}_tree.png"
            stats_png = output_dir / f"{base_name}_stats.png"
            
            # ツリー図作成
            visualizer.visualize_tree(tree_png, figsize=(24, 14), dpi=150)
            
            # 統計プロット作成
            visualizer.create_statistics_plot(stats_png)
            
            print(f"✅ [{i}/{len(pkl_files)}] 完了\n")
            
        except Exception as e:
            print(f"❌ エラー: {pkl_file.name} の処理中に問題が発生しました")
            print(f"   詳細: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "=" * 80)
    print("✅ 全ての可視化が完了しました")
    print(f"📂 出力先: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    # デフォルトパス設定
    pkl_dir = r"C:\Users\yasun\LangChain\learning-langchain\multimodal-raptor-colvbert-blip\data\encoder_comparison_46pdfs\results"
    output_dir = r"C:\Users\yasun\LangChain\learning-langchain\multimodal-raptor-colvbert-blip\data\encoder_comparison_46pdfs\raptor_trees"
    
    # 全ツリーを可視化（形態素解析モード: MeCabインストール済み）
    # use_morphology=True: MeCab形態素解析で名詞・固有名詞抽出、災害語彙優先
    # top_n=2: 各ノードに2つのキーワードを表示（厳密な階層的除外）
    visualize_all_trees(
        pkl_dir=pkl_dir,
        output_dir=output_dir,
        pattern="scaling_test_tree_*.pkl",  # 全チャンクサイズ（2000, 3000など）
        use_morphology=True,  # 形態素解析ON: MeCab + UniDic体言止め形式
        top_n=2  # 2つのキーワードを表示（階層的重複を厳密に排除）
    )
