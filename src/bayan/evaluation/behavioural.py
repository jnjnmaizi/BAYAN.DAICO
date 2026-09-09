"""Run explicit topic invariance and minimum-functionality checks."""
import pandas as pd
from bayan.models.training import ROOT

MFT = [
 ('roads','There is a dangerous pothole in the road.'),('roads','الطريق يحتاج إصلاح الحفر والأسفلت'),
 ('lighting','The street lights are broken.'),('lighting','إنارة الشارع لا تعمل والشارع مظلم'),
 ('waste','The garbage bin is overflowing and needs collection.'),('waste','الحاوية ممتلئة ولم يتم جمع النفايات'),
 ('water','There is a water leak and no water supply.'),('water','انقطاع المياه وتسرب في أنبوب المياه'),
 ('billing','I paid the bill but it still shows as unpaid.'),('billing','دفعت الفاتورة لكن الحالة ما زالت غير مسددة'),
 ('licensing','My permit renewal application is delayed.'),('licensing','طلب تجديد الرخصة متأخر'),
 ('parks','The playground in the park needs maintenance.'),('parks','ألعاب الحديقة تحتاج صيانة'),
 ('digital_services','The online portal crashes when I log in.'),('digital_services','التطبيق يتعطل عند تسجيل الدخول'),
]


def run_behavioural_suite(predict, sentiment_score=None):
    templates=pd.read_csv(ROOT/'data/eval/behavioural_templates.csv')
    originals=[]; variants=[]; identifiers=[]
    for row in templates[templates.test_type=='invariance'].to_dict('records'):
        text=row['template'].format(term=row['term'])
        originals.append(text); variants.append('  '+text.replace(' ','   ')+'  '); identifiers.append(row['test_id'])
    before=predict(originals);after=predict(variants)
    inv=[{'test_id':i,'original':a,'variant':b,'prediction_before':p,'prediction_after':q,'passed':p==q} for i,a,b,p,q in zip(identifiers,originals,variants,before,after)]
    preds=predict([text for _,text in MFT])
    mft=[{'text':text,'expected':gold,'prediction':pred,'passed':gold==pred} for (gold,text),pred in zip(MFT,preds)]
    directional=[]
    for row in templates[templates.test_type=='directional'].to_dict('records'):
        negative=row['template'].format(term=row['term'])
        positive=negative.replace('not working','working').replace('لا تعمل','تعمل')
        if sentiment_score is None:
            directional.append({'test_id':row['test_id'],'status':'not_applicable','reason':'Topic model has no sentiment output; cannot grade sentiment direction using topic probabilities'})
        else:
            directional.append({'test_id':row['test_id'],'passed':sentiment_score(negative)<=sentiment_score(positive)})
    return {'invariance':{'passed':sum(r['passed'] for r in inv),'total':len(inv),'rate':sum(r['passed'] for r in inv)/len(inv),'checks':inv},
        'mft':{'passed':sum(r['passed'] for r in mft),'total':len(mft),'rate':sum(r['passed'] for r in mft)/len(mft),'checks':mft},
        'directional':{'total':len(directional),'assessable':sentiment_score is not None,'checks':directional},
        'limitations':'Invariance checks supplied templates under whitespace changes only; MFT cases are explicit supplemental development probes, not held-out accuracy.'}
