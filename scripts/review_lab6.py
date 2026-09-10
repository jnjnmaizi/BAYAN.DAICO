"""Save/resume the course's per-example human review; no automatic confirmations."""
import argparse
import json
from collections import Counter
from pathlib import Path
from bayan.models.training import ROOT

DEFAULT_REVIEW = ROOT/'artifacts/lab6/human_error_review.json'
DEFAULT_ASSISTANT = ROOT/'artifacts/lab6/assistant_error_review.json'


def is_confirmed(row):
    return row.get('human_confirmed') is True and bool(str(row.get('category') or '').strip()) and bool(str(row.get('reviewer_note') or '').strip())


def save_review(path, rows):
    # Replace atomically so interruption does not leave half-written JSON.
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(rows,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def review_entries(path, suggestions, *, limit=5, input_fn=input, output_fn=print):
    if limit<1:raise ValueError('Review limit must be positive')
    rows=json.loads(path.read_text()); completed=0
    for row in rows:
        if is_confirmed(row):continue
        output_fn(f"\n{row['feedback_id']} | Human-confirmed {sum(is_confirmed(r) for r in rows)}/{len(rows)}")
        output_fn(row['text'])
        output_fn(f"Expected topic: {row['y_true']} | Supplied prediction: {row['y_pred']}")
        hint=suggestions.get(row['feedback_id'])
        if hint and any(hint.get(k)!=row.get(k) for k in ['text','y_true','y_pred']):
            hint=None
            output_fn('Assistant suggestion is stale; review this entry independently.')
        if hint:
            output_fn('Assistant suggestion (not human-confirmed): '+hint['assistant_category'])
            output_fn(hint['assistant_note'])
        action=input_fn('[a] I read this entry and agree with the suggestion; [e] enter my assessment; [s] skip; [q] save and quit: ').strip().lower()
        if action=='q':break
        if action=='s':continue
        if action=='a' and hint:
            category=hint['assistant_category']; note=hint['assistant_note']
            method='human explicitly adopted the displayed assistant assessment after reading the entry'
        elif action=='e':
            category=input_fn('Your error category: ').strip()
            note=input_fn('Your explanation based on this text: ').strip()
            method='human entered an assessment'
        else:
            output_fn('No decision recorded.');continue
        if not category or not note:
            output_fn('Category and explanation are required; entry remains pending.');continue
        row.update(category=category,reviewer_note=note,human_confirmed=True,review_method=method)
        save_review(path,rows);completed+=1
        output_fn('Saved this entry.')
        if completed>=limit:break
    output_fn(f"Session saved {completed} decisions. Total human-confirmed: {sum(is_confirmed(r) for r in rows)}/{len(rows)}.")
    return completed


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit',type=int,default=5,help='Stop after this many explicit decisions; default 5')
    parser.add_argument('--status',action='store_true',help='Show progress without changing any review fields')
    args=parser.parse_args()
    if args.status:
        rows=json.loads(DEFAULT_REVIEW.read_text());confirmed=[r for r in rows if is_confirmed(r)]
        print(json.dumps({'human_confirmed':len(confirmed),'required':len(rows),
            'pending':len(rows)-len(confirmed),'histogram':dict(Counter(r['category'] for r in confirmed))},indent=2));return
    suggestions=json.loads(DEFAULT_ASSISTANT.read_text()) if DEFAULT_ASSISTANT.exists() else []
    try:
        review_entries(DEFAULT_REVIEW,{r['feedback_id']:r for r in suggestions},limit=args.limit)
    except (KeyboardInterrupt,EOFError):
        print('\nStopped. Earlier saved decisions remain available; the current unanswered entry is pending.')


if __name__=='__main__':main()
