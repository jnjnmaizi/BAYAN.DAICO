# Lab 6 — Short review worksheet (Filled)

**Historical status: 0/120 human-confirmed.** This retained submission predates the completed individual review.

The 120 sampled entries are grouped into **45 exact text/expected-label/prediction combinations**. Write your decision and notes once beside each group. All original IDs remain listed; no entries or labels have been removed. This worksheet reduces repeated writing, not the course requirement. Group decisions are not automatically propagated to human confirmations.

## How to use this file

For each group: read the text, compare the expected and predicted topics, then fill in **Decision** and **Your notes**. Use Agree / Disagree / Unsure. You may write notes in Arabic or English.

The first section highlights taxonomy or context questions worth closer attention. These are qualitative priorities, **not calibrated confidence scores**. The model’s stored confidence is confidence in its wrong prediction, not confidence in the proposed review. No 85% review-confidence filter has been applied.

## Start here — context or taxonomy questions

### G01 — 1 sampled entries

> الري متوقف في حديقة حي العليا <PHONE>

Expected: **parks** · Model predicted: **roads**

Proposed assessment: The text reports stopped irrigation within a named park. The explicit park asset supports the supplied parks label; roads is unsupported. Service ownership could distinguish parks/water in another taxonomy, but it does not justify roads here.

Review focus: Check service ownership: park irrigation could be assigned to parks or water under different taxonomies; the supplied label is parks.

- **Decision:** Agree
- **Your notes:** الري داخل حديقة محددة، فالتصنيف الصحيح parks. توقف الري قد يخص جهة المياه أيضاً في تصنيفات أخرى، لكن هذا لا يجعله roads.

Source IDs: FB-000408

### G02 — 4 sampled entries

> الري متوقف في حديقة الدمام

Expected: **parks** · Model predicted: **roads**

Proposed assessment: The text reports stopped irrigation within a named park. The explicit park asset supports the supplied parks label; roads is unsupported. Service ownership could distinguish parks/water in another taxonomy, but it does not justify roads here.

Review focus: Check service ownership: park irrigation could be assigned to parks or water under different taxonomies; the supplied label is parks.

- **Decision:** Agree
- **Your notes:** نفس الحالة السابقة، الأصل حديقة لا طريق.

Source IDs: FB-000688, FB-006568, FB-011128, FB-001208

### G03 — 1 sampled entries

> لووووسمحت ألعاب الأطفال في حديقة شارع التحلية تحتاج صيانة

Expected: **parks** · Model predicted: **roads**

Proposed assessment: The text requests maintenance of children's play equipment inside a park. The named asset supports parks; the supplied roads prediction does not match it. The elongated polite prefix adds surface noise without changing the requested service. The street/road word belongs to the place name after حديقة, not proof that the complaint concerns roads.

Review focus: Check that the road/street wording names the park location rather than the requested service.

- **Decision:** Agree
- **Your notes:** كلمة "شارع" هنا جزء من اسم موقع الحديقة، والخدمة المطلوبة صيانة ألعاب أطفال داخل حديقة، فالتصنيف parks صحيح.

Source IDs: FB-001248

### G04 — 5 sampled entries

> الممر في حديقة شارع التحلية غير مناسب للكراسي المتحركة

Expected: **parks** · Model predicted: **roads**

Proposed assessment: The text concerns wheelchair access on a walkway explicitly inside a park. The park context supports parks rather than a public-road complaint. The street/road word belongs to the place name after حديقة, not proof that the complaint concerns roads.

Review focus: Check that the road/street wording names the park location rather than the requested service.

- **Decision:** Agree
- **Your notes:** الممر المذكور داخل الحديقة نفسها، وليس طريقاً عاماً، فـ parks هو التصنيف الصحيح.

Source IDs: FB-002928, FB-009248, FB-006528, FB-006168, FB-000648

### G05 — 1 sampled entries

> الري متوقف في حديقة الدمام  

Expected: **parks** · Model predicted: **roads**

Proposed assessment: The text reports stopped irrigation within a named park. The explicit park asset supports the supplied parks label; roads is unsupported. Service ownership could distinguish parks/water in another taxonomy, but it does not justify roads here.

Review focus: Check service ownership: park irrigation could be assigned to parks or water under different taxonomies; the supplied label is parks.

- **Decision:** Agree
- **Your notes:** تكرار لنفس نص G02، والحكم نفسه: parks صحيح.

Source IDs: FB-003008

### G06 — 1 sampled entries

> ألري متوقف في حديقة الدمام

Expected: **parks** · Model predicted: **roads**

Proposed assessment: The text reports stopped irrigation within a named park. The explicit park asset supports the supplied parks label; roads is unsupported. Service ownership could distinguish parks/water in another taxonomy, but it does not justify roads here. A visible hamza/spelling variant is present, but the park context remains explicit; its causal effect is untested.

Review focus: Check service ownership: park irrigation could be assigned to parks or water under different taxonomies; the supplied label is parks.

- **Decision:** Agree
- **Your notes:** اختلاف الهمزة (ألري/الري) لا يغيّر المعنى أو التصنيف؛ يبقى parks.

Source IDs: FB-003288

### G07 — 4 sampled entries

> الري متوقف في حديقة حي العليا

Expected: **parks** · Model predicted: **roads**

Proposed assessment: The text reports stopped irrigation within a named park. The explicit park asset supports the supplied parks label; roads is unsupported. Service ownership could distinguish parks/water in another taxonomy, but it does not justify roads here.

Review focus: Check service ownership: park irrigation could be assigned to parks or water under different taxonomies; the supplied label is parks.

- **Decision:** Agree
- **Your notes:** نفس نمط G01، حديقة صريحة داخل حي معيّن.

Source IDs: FB-004448, FB-004328, FB-005608, FB-008248

### G08 — 1 sampled entries

> لووووسمحت الممر في حديقة طريق الملك فهد غير مناسب للكراسي المتحركة

Expected: **parks** · Model predicted: **roads**

Proposed assessment: The text concerns wheelchair access on a walkway explicitly inside a park. The park context supports parks rather than a public-road complaint. The elongated polite prefix adds surface noise without changing the requested service. The street/road word belongs to the place name after حديقة, not proof that the complaint concerns roads.

Review focus: Check that the road/street wording names the park location rather than the requested service.

- **Decision:** Agree
- **Your notes:** "طريق الملك فهد" اسم موقع الحديقة فقط، والممر المتحدث عنه داخلها، فالتصنيف parks صحيح.

Source IDs: FB-005888

### G09 — 7 sampled entries

> الري متوقف في حديقة الرياض

Expected: **parks** · Model predicted: **roads**

Proposed assessment: The text reports stopped irrigation within a named park. The explicit park asset supports the supplied parks label; roads is unsupported. Service ownership could distinguish parks/water in another taxonomy, but it does not justify roads here.

Review focus: Check service ownership: park irrigation could be assigned to parks or water under different taxonomies; the supplied label is parks.

- **Decision:** Agree
- **Your notes:** نفس النمط، حديقة الرياض اسم مكان محدد، parks صحيح.

Source IDs: FB-006088, FB-011248, FB-003088, FB-002728, FB-010968, FB-010488, FB-009928

### G10 — 4 sampled entries

> الري متوقف في حديقة جدة

Expected: **parks** · Model predicted: **roads**

Proposed assessment: The text reports stopped irrigation within a named park. The explicit park asset supports the supplied parks label; roads is unsupported. Service ownership could distinguish parks/water in another taxonomy, but it does not justify roads here.

Review focus: Check service ownership: park irrigation could be assigned to parks or water under different taxonomies; the supplied label is parks.

- **Decision:** Agree
- **Your notes:** نفس النمط المتكرر، parks صحيح.

Source IDs: FB-006608, FB-009528, FB-002368, FB-005848

### G11 — 2 sampled entries

> ألعاب الأطفال في حديقة شارع التحلية تحتاج صيانة 😡

Expected: **parks** · Model predicted: **roads**

Proposed assessment: The text requests maintenance of children's play equipment inside a park. The named asset supports parks; the supplied roads prediction does not match it. The street/road word belongs to the place name after حديقة, not proof that the complaint concerns roads.

Review focus: Check that the road/street wording names the park location rather than the requested service.

- **Decision:** Agree
- **Your notes:** الإيموجي لا يغيّر التصنيف؛ الطلب صيانة ألعاب أطفال داخل حديقة.

Source IDs: FB-007008, FB-001808

### G12 — 2 sampled entries

> الري متوقف في حديقة طريق الملك فهد

Expected: **parks** · Model predicted: **roads**

Proposed assessment: The text reports stopped irrigation within a named park. The explicit park asset supports the supplied parks label; roads is unsupported. Service ownership could distinguish parks/water in another taxonomy, but it does not justify roads here. The street/road word belongs to the place name after حديقة, not proof that the complaint concerns roads.

Review focus: Check service ownership: park irrigation could be assigned to parks or water under different taxonomies; the supplied label is parks. Check that the road/street wording names the park location rather than the requested service.

- **Decision:** Agree
- **Your notes:** "طريق الملك فهد" جزء من اسم الحديقة، والري متوقف داخلها، parks صحيح.

Source IDs: FB-007448, FB-000608

### G13 — 1 sampled entries

> ألعأب الأطفال في حديقة شارع التحلية تحتاج صيانة

Expected: **parks** · Model predicted: **roads**

Proposed assessment: The text requests maintenance of children's play equipment inside a park. The named asset supports parks; the supplied roads prediction does not match it. A visible hamza/spelling variant is present, but the park context remains explicit; its causal effect is untested. The street/road word belongs to the place name after حديقة, not proof that the complaint concerns roads.

Review focus: Check that the road/street wording names the park location rather than the requested service.

- **Decision:** Agree
- **Your notes:** خطأ إملائي بسيط (ألعأب) لا يغيّر المعنى، parks صحيح.

Source IDs: FB-008608

### G14 — 4 sampled entries

> الري متوقف في حديقة حي النرجس

Expected: **parks** · Model predicted: **roads**

Proposed assessment: The text reports stopped irrigation within a named park. The explicit park asset supports the supplied parks label; roads is unsupported. Service ownership could distinguish parks/water in another taxonomy, but it does not justify roads here.

Review focus: Check service ownership: park irrigation could be assigned to parks or water under different taxonomies; the supplied label is parks.

- **Decision:** Agree
- **Your notes:** نفس نمط الري داخل حديقة حي معيّن، parks صحيح.

Source IDs: FB-008728, FB-008448, FB-001688, FB-010248

### G15 — 1 sampled entries

> ألعاب الأطفال في حديقة شارع التحلية تحتاج صيانة <PHONE>

Expected: **parks** · Model predicted: **roads**

Proposed assessment: The text requests maintenance of children's play equipment inside a park. The named asset supports parks; the supplied roads prediction does not match it. The street/road word belongs to the place name after حديقة, not proof that the complaint concerns roads.

Review focus: Check that the road/street wording names the park location rather than the requested service.

- **Decision:** Agree
- **Your notes:** رقم الهاتف لا يؤثر على التصنيف، parks صحيح.

Source IDs: FB-009288

### G16 — 4 sampled entries

> ألعاب الأطفال في حديقة شارع التحلية تحتاج صيانة

Expected: **parks** · Model predicted: **roads**

Proposed assessment: The text requests maintenance of children's play equipment inside a park. The named asset supports parks; the supplied roads prediction does not match it. The street/road word belongs to the place name after حديقة, not proof that the complaint concerns roads.

Review focus: Check that the road/street wording names the park location rather than the requested service.

- **Decision:** Agree
- **Your notes:** نفس نمط G03/G11، parks صحيح.

Source IDs: FB-009488, FB-002288, FB-007288, FB-007408

### G17 — 4 sampled entries

> ألعاب الأطفال في حديقة طريق الملك فهد تحتاج صيانة

Expected: **parks** · Model predicted: **roads**

Proposed assessment: The text requests maintenance of children's play equipment inside a park. The named asset supports parks; the supplied roads prediction does not match it. The street/road word belongs to the place name after حديقة, not proof that the complaint concerns roads.

Review focus: Check that the road/street wording names the park location rather than the requested service.

- **Decision:** Agree
- **Your notes:** اسم الطريق هنا جزء من اسم الحديقة فقط، parks صحيح.

Source IDs: FB-009568, FB-003688, FB-005048, FB-004568

### G18 — 1 sampled entries

> الممر في حديقة طريق الملك فهد غير مناسب للكراسي المتحركة 😡

Expected: **parks** · Model predicted: **roads**

Proposed assessment: The text concerns wheelchair access on a walkway explicitly inside a park. The park context supports parks rather than a public-road complaint. The street/road word belongs to the place name after حديقة, not proof that the complaint concerns roads.

Review focus: Check that the road/street wording names the park location rather than the requested service.

- **Decision:** Agree
- **Your notes:** الممر داخل الحديقة، parks صحيح.

Source IDs: FB-010648

### G19 — 4 sampled entries

> الري متوقف في حديقة شارع التحلية

Expected: **parks** · Model predicted: **roads**

Proposed assessment: The text reports stopped irrigation within a named park. The explicit park asset supports the supplied parks label; roads is unsupported. Service ownership could distinguish parks/water in another taxonomy, but it does not justify roads here. The street/road word belongs to the place name after حديقة, not proof that the complaint concerns roads.

Review focus: Check service ownership: park irrigation could be assigned to parks or water under different taxonomies; the supplied label is parks. Check that the road/street wording names the park location rather than the requested service.

- **Decision:** Agree
- **Your notes:** نفس نمط الري داخل حديقة تحمل اسم شارع، parks صحيح.

Source IDs: FB-011568, FB-001848, FB-000888, FB-005288

### G20 — 7 sampled entries

> الممر في حديقة طريق الملك فهد غير مناسب للكراسي المتحركة

Expected: **parks** · Model predicted: **roads**

Proposed assessment: The text concerns wheelchair access on a walkway explicitly inside a park. The park context supports parks rather than a public-road complaint. The street/road word belongs to the place name after حديقة, not proof that the complaint concerns roads.

Review focus: Check that the road/street wording names the park location rather than the requested service.

- **Decision:** Agree
- **Your notes:** تكرار نفس نمط G08/G18، parks صحيح.

Source IDs: FB-011808, FB-006288, FB-001328, FB-004768, FB-011448, FB-007208, FB-007328

## Remaining groups — direct park context

These still need a real review; they have not been marked correct or excluded.

### G21 — 1 sampled entries

> ألعأب الأطفال في حديقة الدمام تحتاج صيانة 😡

Expected: **parks** · Model predicted: **roads**

Proposed assessment: The text requests maintenance of children's play equipment inside a park. The named asset supports parks; the supplied roads prediction does not match it. A visible hamza/spelling variant is present, but the park context remains explicit; its causal effect is untested.

- **Decision:** Agree
- **Your notes:** لا يوجد ذكر لطريق هنا أصلاً، الحديقة صريحة، parks صحيح بوضوح.

Source IDs: FB-000248

### G22 — 1 sampled entries

> ألعاب الأطفال في حديقة الرياض تحتاج صيانة <NATIONAL_ID>

Expected: **parks** · Model predicted: **roads**

Proposed assessment: The text requests maintenance of children's play equipment inside a park. The named asset supports parks; the supplied roads prediction does not match it.

- **Decision:** Agree
- **Your notes:** نص مباشر عن ألعاب حديقة، لا علاقة بطرق، parks صحيح.

Source IDs: FB-000288

### G23 — 1 sampled entries

> ألعاب الأطفال في حديقة الرياض تحتاج صيانة 😡

Expected: **parks** · Model predicted: **roads**

Proposed assessment: The text requests maintenance of children's play equipment inside a park. The named asset supports parks; the supplied roads prediction does not match it.

- **Decision:** Agree
- **Your notes:** نفس G22، parks صحيح.

Source IDs: FB-000768

### G24 — 3 sampled entries

> الممر في حديقة حي العليا غير مناسب للكراسي المتحركة

Expected: **parks** · Model predicted: **roads**

Proposed assessment: The text concerns wheelchair access on a walkway explicitly inside a park. The park context supports parks rather than a public-road complaint.

- **Decision:** Agree
- **Your notes:** ممر داخل حديقة، لا يوجد ذكر لطريق عام، parks صحيح.

Source IDs: FB-000968, FB-003728, FB-011528

### G25 — 1 sampled entries

> ألممر في حديقة الدمام غير مناسب للكراسي المتحركة

Expected: **parks** · Model predicted: **roads**

Proposed assessment: The text concerns wheelchair access on a walkway explicitly inside a park. The park context supports parks rather than a public-road complaint. A visible hamza/spelling variant is present, but the park context remains explicit; its causal effect is untested.

- **Decision:** Agree
- **Your notes:** اختلاف الهمزة لا يغيّر التصنيف، parks صحيح.

Source IDs: FB-001008

### G26 — 1 sampled entries

> لووووسمحت الممر في حديقة الرياض غير مناسب للكراسي المتحركة

Expected: **parks** · Model predicted: **roads**

Proposed assessment: The text concerns wheelchair access on a walkway explicitly inside a park. The park context supports parks rather than a public-road complaint. The elongated polite prefix adds surface noise without changing the requested service.

- **Decision:** Agree
- **Your notes:** الإطالة في "لووووسمحت" أسلوبية فقط، parks صحيح.

Source IDs: FB-002408

### G27 — 3 sampled entries

> ألعاب الأطفال في حديقة حي النرجس تحتاج صيانة

Expected: **parks** · Model predicted: **roads**

Proposed assessment: The text requests maintenance of children's play equipment inside a park. The named asset supports parks; the supplied roads prediction does not match it.

- **Decision:** Agree
- **Your notes:** لا صلة بالطرق، parks صحيح.

Source IDs: FB-002648, FB-009128, FB-003848

### G28 — 5 sampled entries

> ألعاب الأطفال في حديقة الرياض تحتاج صيانة

Expected: **parks** · Model predicted: **roads**

Proposed assessment: The text requests maintenance of children's play equipment inside a park. The named asset supports parks; the supplied roads prediction does not match it.

- **Decision:** Agree
- **Your notes:** نص واضح عن حديقة، parks صحيح.

Source IDs: FB-003128, FB-007768, FB-008968, FB-007168, FB-000728

### G29 — 1 sampled entries

> ألعاب الأطفال في حديقة حي الياسمين تحتاج صيانة 😡 <PHONE>

Expected: **parks** · Model predicted: **roads**

Proposed assessment: The text requests maintenance of children's play equipment inside a park. The named asset supports parks; the supplied roads prediction does not match it.

- **Decision:** Agree
- **Your notes:** الإيموجي ورقم الهاتف لا يغيّران التصنيف، parks صحيح.

Source IDs: FB-003368

### G30 — 1 sampled entries

> الممر في حديقة حي العليا غير مناسب للكراسي المتحركة 😡

Expected: **parks** · Model predicted: **roads**

Proposed assessment: The text concerns wheelchair access on a walkway explicitly inside a park. The park context supports parks rather than a public-road complaint.

- **Decision:** Agree
- **Your notes:** نفس G24 مع إيموجي إضافي، parks صحيح.

Source IDs: FB-003888

### G31 — 1 sampled entries

> ألعأب الأطفال في حديقة الرياض تحتاج صيانة

Expected: **parks** · Model predicted: **roads**

Proposed assessment: The text requests maintenance of children's play equipment inside a park. The named asset supports parks; the supplied roads prediction does not match it. A visible hamza/spelling variant is present, but the park context remains explicit; its causal effect is untested.

- **Decision:** Agree
- **Your notes:** خطأ إملائي بسيط لا يغيّر التصنيف، parks صحيح.

Source IDs: FB-004048

### G32 — 3 sampled entries

> ألعاب الأطفال في حديقة حي العليا تحتاج صيانة

Expected: **parks** · Model predicted: **roads**

Proposed assessment: The text requests maintenance of children's play equipment inside a park. The named asset supports parks; the supplied roads prediction does not match it.

- **Decision:** Agree
- **Your notes:** نص مباشر، parks صحيح.

Source IDs: FB-004368, FB-004168, FB-002688

### G33 — 1 sampled entries

> لووووسمحت الممر في حديقة جدة غير مناسب للكراسي المتحركة

Expected: **parks** · Model predicted: **roads**

Proposed assessment: The text concerns wheelchair access on a walkway explicitly inside a park. The park context supports parks rather than a public-road complaint. The elongated polite prefix adds surface noise without changing the requested service.

- **Decision:** Agree
- **Your notes:** الإطالة اللفظية لا تغيّر المضمون، parks صحيح.

Source IDs: FB-004728

### G34 — 3 sampled entries

> ألعاب الأطفال في حديقة الدمام تحتاج صيانة

Expected: **parks** · Model predicted: **roads**

Proposed assessment: The text requests maintenance of children's play equipment inside a park. The named asset supports parks; the supplied roads prediction does not match it.

- **Decision:** Agree
- **Your notes:** نفس النمط، parks صحيح.

Source IDs: FB-004968, FB-001488, FB-009808

### G35 — 2 sampled entries

> ألعأب الأطفال في حديقة الدمام تحتاج صيانة

Expected: **parks** · Model predicted: **roads**

Proposed assessment: The text requests maintenance of children's play equipment inside a park. The named asset supports parks; the supplied roads prediction does not match it. A visible hamza/spelling variant is present, but the park context remains explicit; its causal effect is untested.

- **Decision:** Agree
- **Your notes:** خطأ إملائي بسيط، parks صحيح.

Source IDs: FB-007848, FB-002528

### G36 — 8 sampled entries

> الممر في حديقة جدة غير مناسب للكراسي المتحركة

Expected: **parks** · Model predicted: **roads**

Proposed assessment: The text concerns wheelchair access on a walkway explicitly inside a park. The park context supports parks rather than a public-road complaint.

- **Decision:** Agree
- **Your notes:** ممر داخل حديقة جدة تحديداً، parks صحيح.

Source IDs: FB-008128, FB-000368, FB-000208, FB-005928, FB-003048, FB-003168, FB-004648, FB-007888

### G37 — 4 sampled entries

> الممر في حديقة حي النرجس غير مناسب للكراسي المتحركة

Expected: **parks** · Model predicted: **roads**

Proposed assessment: The text concerns wheelchair access on a walkway explicitly inside a park. The park context supports parks rather than a public-road complaint.

- **Decision:** Agree
- **Your notes:** نفس النمط، parks صحيح.

Source IDs: FB-008848, FB-006928, FB-009008, FB-007248

### G38 — 4 sampled entries

> الممر في حديقة الدمام غير مناسب للكراسي المتحركة

Expected: **parks** · Model predicted: **roads**

Proposed assessment: The text concerns wheelchair access on a walkway explicitly inside a park. The park context supports parks rather than a public-road complaint.

- **Decision:** Agree
- **Your notes:** نفس النمط، parks صحيح.

Source IDs: FB-009048, FB-004528, FB-002248, FB-002768

### G39 — 2 sampled entries

> الممر في حديقة حي الياسمين غير مناسب للكراسي المتحركة

Expected: **parks** · Model predicted: **roads**

Proposed assessment: The text concerns wheelchair access on a walkway explicitly inside a park. The park context supports parks rather than a public-road complaint.

- **Decision:** Agree
- **Your notes:** نفس النمط، parks صحيح.

Source IDs: FB-009168, FB-002888

### G40 — 1 sampled entries

> الممر في حديقة جدة غير مناسب للكراسي المتحركة  

Expected: **parks** · Model predicted: **roads**

Proposed assessment: The text concerns wheelchair access on a walkway explicitly inside a park. The park context supports parks rather than a public-road complaint.

- **Decision:** Agree
- **Your notes:** تكرار لنص G36، parks صحيح.

Source IDs: FB-009208

### G41 — 3 sampled entries

> ألعاب الأطفال في حديقة جدة تحتاج صيانة

Expected: **parks** · Model predicted: **roads**

Proposed assessment: The text requests maintenance of children's play equipment inside a park. The named asset supports parks; the supplied roads prediction does not match it.

- **Decision:** Agree
- **Your notes:** نص مباشر عن حديقة جدة، parks صحيح.

Source IDs: FB-009328, FB-010008, FB-007688

### G42 — 1 sampled entries

> لووووسمحت ألممر في حديقة حي العليا غير مناسب للكراسي المتحركة

Expected: **parks** · Model predicted: **roads**

Proposed assessment: The text concerns wheelchair access on a walkway explicitly inside a park. The park context supports parks rather than a public-road complaint. A visible hamza/spelling variant is present, but the park context remains explicit; its causal effect is untested. The elongated polite prefix adds surface noise without changing the requested service.

- **Decision:** Agree
- **Your notes:** لا تأثير للإطالة أو الهمزة على المعنى، parks صحيح.

Source IDs: FB-009368

### G43 — 5 sampled entries

> ألعاب الأطفال في حديقة حي الياسمين تحتاج صيانة

Expected: **parks** · Model predicted: **roads**

Proposed assessment: The text requests maintenance of children's play equipment inside a park. The named asset supports parks; the supplied roads prediction does not match it.

- **Decision:** Agree
- **Your notes:** نص مباشر عن حديقة، parks صحيح.

Source IDs: FB-009968, FB-003608, FB-005768, FB-007728, FB-005168

### G44 — 1 sampled entries

> لووووسمحت الممر في حديقة الدمام غير مناسب للكراسي المتحركة 😡  

Expected: **parks** · Model predicted: **roads**

Proposed assessment: The text concerns wheelchair access on a walkway explicitly inside a park. The park context supports parks rather than a public-road complaint. The elongated polite prefix adds surface noise without changing the requested service.

- **Decision:** Agree
- **Your notes:** الإطالة والإيموجي لا يغيّران التصنيف، parks صحيح.

Source IDs: FB-011688

### G45 — 4 sampled entries

> الممر في حديقة الرياض غير مناسب للكراسي المتحركة

Expected: **parks** · Model predicted: **roads**

Proposed assessment: The text concerns wheelchair access on a walkway explicitly inside a park. The park context supports parks rather than a public-road complaint.

- **Decision:** Agree
- **Your notes:** نفس النمط، parks صحيح.

Source IDs: FB-011888, FB-011288, FB-010608, FB-003448
