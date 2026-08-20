import re, math

def _norm(s): return (s or '').lower().replace('ё','е').replace('–','-').replace('—','-')
def _txt(opts, needles):
    for k,v in opts.items():
        s=_norm(v)
        if any(_norm(n) in s for n in needles): return k
    return None
def _num(opts, value, tol=1e-8):
    for k,v in opts.items():
        s=(v or '').replace('°','').replace('%','').replace(',','.').strip()
        if re.fullmatch(r'-?\d+(?:\.\d+)?',s) and abs(float(s)-float(value))<=tol:return k
    return None

def solve(item):
    q=_norm(item['q']); opts=item['options']
    # Exact arithmetic / elementary math.
    m=re.search(r'(?:чему равно|вычисли|найдите?)\s+(\d+)\s*[×*]\s*(\d+)',q)
    if m:return _num(opts,int(m[1])*int(m[2])),'math_mul'
    m=re.search(r'(?:чему равно|вычисли|найдите?)\s+(\d+)\s*\+\s*(\d+)',q)
    if m:return _num(opts,int(m[1])+int(m[2])),'math_add'
    m=re.search(r'(?:чему равно|вычисли|найдите?)\s+(\d+)\s*-\s*(\d+)',q)
    if m:return _num(opts,int(m[1])-int(m[2])),'math_sub'
    m=re.search(r'(?:чему равно|вычисли|найдите?)\s+(\d+)\s*/\s*(\d+)',q)
    if m and int(m[2]):return _num(opts,int(m[1])/int(m[2])),'math_div'
    m=re.search(r'наибольш(?:ий|его) общ(?:ий|его) делител[ья].*?(\d+).*?(\d+)',q)
    if m:return _num(opts,math.gcd(int(m[1]),int(m[2]))),'math_gcd'
    m=re.search(r'x\s*/\s*(\d+)\s*=\s*(\d+)',q)
    if m:return _num(opts,int(m[1])*int(m[2])),'linear_equation_div'
    m=re.search(r'(\d+)\s*x\s*=\s*(\d+)',q)
    if m and int(m[1]):return _num(opts,int(m[2])/int(m[1])),'linear_equation_mul'
    if 'сумм' in q and 'внутрен' in q and 'угл' in q and 'треуголь' in q:return _num(opts,180),'triangle_angle_sum'
    if 'медиан' in q:
        nums=[int(x) for x in re.findall(r'\b\d+\b',q)]
        if len(nums)>=3:
            v=sorted(nums[-5:] if len(nums)>=5 else nums);return _num(opts,v[len(v)//2]),'median'
    m=re.search(r'(?:сколько процентов составляет|какой процент составляет)\s+(\d+)\s+от\s+(\d+)',q)
    if m and int(m[2]):return _num(opts,100*int(m[1])/int(m[2])),'percent_of'
    m=re.search(r'(?:чему равно|вычисли)\s+(\d+)\s*\^\s*(\d+)',q)
    if m:return _num(opts,int(m[1])**int(m[2])),'power'
    m=re.search(r'прямоугольник.*сторон\w*\s+(\d+)\s+и\s+(\d+).*площад',q)
    if m:return _num(opts,int(m[1])*int(m[2])),'rectangle_area'
    m=re.search(r'периметр.*прямоугольник.*сторон\w*\s+(\d+)\s+и\s+(\d+)',q)
    if m:return _num(opts,2*(int(m[1])+int(m[2]))),'rectangle_perimeter'
    m=re.search(r'резерв\w*\s+(\d+).*потер\w*\s+(\d+).*пополн\w*\s+(\d+)',q)
    if m:return _num(opts,int(m[1])-int(m[2])+int(m[3])),'balance_arithmetic'
    m=re.search(r'участк\w*.*?(\d+)\s*км.*?(\d+)\s*км.*перв\w*.*процент',q)
    if m:
        a,b=map(int,m.groups());return _num(opts,100*a/(a+b)),'route_percent'
    if 'математическ' in q and 'ожидан' in q and 'независим' in q:
        vals=[float(x.replace(',','.')) for x in re.findall(r'\d+(?:[.,]\d+)?',q)]
        p=next((x for x in vals if 0<=x<=1),None);n=next((int(x) for x in vals if x>=2 and float(x).is_integer()),None)
        if p is not None and n is not None:return _num(opts,p*n),'binomial_expectation'
    # Formal logic.
    if 'все r являются s' in q and 'ни один s не является t' in q:return _txt(opts,['нет']),'syllogism_no_r_t'
    if 'если p истинно, то q истинно' in q and re.search(r'\bp истинно',q):return _txt(opts,['q истинно']),'modus_ponens'
    if ('все a - b' in q or 'все a — b' in q) and 'обязательно' in q:return _txt(opts,['ни один a не является не-b','ни один a не является не b']),'universal_implication'
    if 'взаимоисключ' in q and ('x произошло' in q or 'событие x произошло' in q):return _txt(opts,['y не произошло']),'mutual_exclusion'
    if 'a > b' in q and 'b > c' in q:return _txt(opts,['a > c']),'order_transitivity'
    if 'де морган' in q and ('не (p и q)' in q or 'not (p and q)' in q):return _txt(opts,['не p или не q','not p or not q']),'de_morgan'
    if 'ровно одно' in q and 'a' in q and 'b' in q and 'a ложно' in q:return _txt(opts,['b истинно']),'xor'
    if 'modus tollens' in q or ('если идет дождь' in q and ('улица не мокрая' in q or 'дорога не мокрая' in q)):return _txt(opts,['дождя нет','дождь не идет']),'modus_tollens'
    if 'если a, то b' in q and 'b ложно' in q:return _txt(opts,['a ложно','a не истинно']),'generic_modus_tollens'
    # Cybernetic / causal invariants.
    if ('увеличивает охлаждение' in q and 'температур' in q) or ('уменьшает нагрев' in q and 'выше' in q):return _txt(opts,['отрицательная обратная связь','negative feedback']),'negative_feedback'
    if ('альтернативн' in q and 'причин' in q) or ('внешн' in q and 'сбой' in q and 'причин' in q):return _txt(opts,['проверить альтернативные причины','контроль','не доказана']),'causal_control'
    if 'одного физического сенсора' in q or ('общ' in q and 'сенсор' in q and 'независим' in q):return _txt(opts,['нет','не являются']),'shared_source_not_independent'
    if 'не измеряет выход' in q or 'нет измерения выхода' in q:return _txt(opts,['открытый']),'open_loop'
    if ('скрыт' in q or 'конфаун' in q) and ('влияет на оба' in q or 'общая причина' in q):return _txt(opts,['причинность не доказана','конфаундер','нельзя заключить']),'confounder'
    if 'накоплен' in q and 'ошиб' in q and 'огранич' in q:return _txt(opts,['обратная связь по ошибке','feedback']),'error_feedback'
    if 'верификатор не подтверждает' in q or 'проверка не подтверждает' in q:return _txt(opts,['не продвигать','не принимать','не довер']),'trust_gate'
    if 'ошибк' in q and 'цель' in q and 'наблюдаем' in q and ('меняет действие' in q or 'корректирует' in q):return _txt(opts,['feedback control','обратная связь']),'feedback_control'
    if 'формальн' in q and ('провер' in q or 'доказ' in q) and ('приоритет' in q or 'выше' in q):return _txt(opts,['формально провер','верифицирован']),'authority_priority'
    if 'криптографическ' in q and ('хэш' in q or 'хеш' in q) and 'не совп' in q:return _txt(opts,['непроверенным','отклонить','reject']),'hash_authority'
    if 'verified=true' in q and ('risk=high' in q or 'risk = high' in q):return _txt(opts,['нет','не разрешена']),'policy_and_gate'
    if 'требует успешного b' in q and ('b не выполнен' in q or 'b завершился ошибкой' in q):return _txt(opts,['нет','нельзя']),'dependency_gate'
    if 'хотя бы два' in q and 'сервер' in q and 'доступ' in q:
        yes=len(re.findall(r'\b[a-zа-я]\s+доступен',q))
        if yes>=2:return _txt(opts,['да']),'quorum'
    # Exact code / CS.
    m=re.search(r'len\(\[([^\]]*)\]\)',q)
    if m:return _num(opts,len([x for x in m[1].split(',') if x.strip()])),'python_len'
    m=re.search(r'python.*?(\d+)\s*%\s*(\d+)',q)
    if m:return _num(opts,int(m[1])%int(m[2])),'python_mod'
    if 'sql' in q and 'фильтр' in q and ('до группиров' in q or 'до group by' in q):return _txt(opts,['where']),'sql_where'
    if ('проход' in q or 'просмотр' in q) and 'всех n' in q and 'элемент' in q:return _txt(opts,['o(n)']),'linear_scan'
    if 'a and b' in q and ('булев' in q or 'логическ' in q):return _txt(opts,['только если истинны оба','оба истинны']),'boolean_and'
    if 'lifo' in q:return _txt(opts,['стек']),'stack_lifo'
    if 'перв' in q and 'элемент' in q and 'списк' in q and 'python' in q:return _num(opts,0),'zero_index'
    if 'break' in q and 'цикл' in q:return _txt(opts,['завершает текущий цикл','выходит из текущего цикла']),'break_loop'
    # Bounded Russian Authority.
    if 'благодаря' in q:
        for n in ['быстрой работе','помощи','поддержке','совету']:
            a=_txt(opts,[n])
            if a:return a,'ru_blagodarya_dative'
    if 'антоним' in q and 'временн' in q:return _txt(opts,['постоянн']),'ru_antonym_temporary'
    if 'хотя' in q and ('союз' in q or 'отношение' in q):return _txt(opts,['уступк']),'ru_concession'
    if 'в течение' in q:
        for n in ['недели','месяца','года','дня']:
            a=_txt(opts,[n])
            if a:return a,'ru_v_techenie_genitive'
    if 'принять во внимание' in q:return _txt(opts,['учесть']),'ru_idiom_attention'
    if any(x in q for x in ['пять','шесть','семь','восемь','девять','десять']):
        for n in ['килограммов','метров','рублей','дней','файлов','серверов','задач','книг']:
            a=_txt(opts,[n])
            if a:return a,'ru_numeral_genitive_plural'
    return None,None
