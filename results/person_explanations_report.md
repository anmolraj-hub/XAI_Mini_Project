# Person-Level GNNExplanations for AIFB

This report explains the trained FastRGCN model in presentation-ready language. Each person section states the model prediction and the graph relations that GNNExplainer considered most important.

## Global Explanation

Across the sampled training nodes, these relation types had the highest average importance:

1. **fax** - importance 0.0926
2. **type** - importance 0.0699
3. **phone** - importance 0.0481
4. **allValuesFrom** - importance 0.0345
5. **head_inverse** - importance 0.0218
6. **subClassOf_inverse** - importance 0.0106
7. **carriedOutBy** - importance 0.0096
8. **carriesOut_inverse** - importance 0.0096
9. **homepage** - importance 0.0080
10. **isWorkedOnBy_inverse** - importance 0.0079

## Local Person Explanations

### Andreas Oberweis (node 5697)

- Predicted group: **Business Information & Communication Systems**
- Model confidence: **1.000**
- Ground-truth match: **True**
- Edges above importance threshold 0.5 (within 2-hop receptive field): **2**
- Table and figure show the top 10 edges by importance

For Andreas Oberweis, the model predicts the research group Business Information & Communication Systems. The explanation says this decision is mainly supported by graph connections of type inverse of fax, head, inverse of type. The strongest single piece of evidence is the connection '+49 (721) 608 4548' -- inverse of fax --> 'Andreas Oberweis' with importance 0.88. In simple terms, the model is looking at this person's surrounding publications, projects, topics, and related entities, then using the most influential links around that person to justify the group prediction.

| Rank | Source | Relation | Target | Importance |
|---:|---|---|---|---:|
| 1 | +49 (721) 608 4548 | fax_inverse | Andreas Oberweis | 0.883 |
| 2 | Business Information and Communication Systems | head | Andreas Oberweis | 0.851 |
| 3 | Research Group | type_inverse | Business Information and Communication Systems | 0.183 |
| 4 | Business Information and Communication Systems | name_inverse | Business Information and Communication Systems | 0.172 |
| 5 | BIK | homepage_inverse | Business Information and Communication Systems | 0.167 |
| 6 | Person | type_inverse | Andreas Oberweis | 0.157 |
| 7 | Andreas Oberweis | name_inverse | Andreas Oberweis | 0.151 |
| 8 | Kirsten Lenz | isWorkedOnBy_inverse | business process management | 0.150 |
| 9 | Wolffried Stucky | publication | Lernobjekte im E-Learning - Eine kritische Beurteilung zugrunde liegender Konzepte anhand eines Vergleichs mit komponentenbasierter Software-Entwicklung | 0.149 |
| 10 | Workflow gestütztes Lernobjekt Management | title_inverse | Workflow-gestütztes Lernobjekt-Management | 0.149 |

### Maik Herfurth (node 5836)

- Predicted group: **Business Information & Communication Systems**
- Model confidence: **1.000**
- Ground-truth match: **True**
- Edges above importance threshold 0.5 (within 2-hop receptive field): **2**
- Table and figure show the top 10 edges by importance

For Maik Herfurth, the model predicts the research group Business Information & Communication Systems. The explanation says this decision is mainly supported by graph connections of type type (3), inverse of fax, member. The strongest single piece of evidence is the connection '+49 (721) 608 4548' -- inverse of fax --> 'Maik Herfurth' with importance 0.84. In simple terms, the model is looking at this person's surrounding publications, projects, topics, and related entities, then using the most influential links around that person to justify the group prediction.

| Rank | Source | Relation | Target | Importance |
|---:|---|---|---|---:|
| 1 | +49 (721) 608 4548 | fax_inverse | Maik Herfurth | 0.836 |
| 2 | Business Information and Communication Systems | member | Maik Herfurth | 0.730 |
| 3 | Petra Haberer | type | Person | 0.145 |
| 4 | K. Aberer | type | Person | 0.145 |
| 5 | Roman Povalej | type | Person | 0.144 |
| 6 | V. Christophides | type | Person | 0.144 |
| 7 | E. Bonabeau | type | Person | 0.144 |
| 8 | Holger Lewen | type | Person | 0.144 |
| 9 | G. Schmidt | type | Person | 0.144 |
| 10 | Günter Ladwig | type | Person | 0.144 |

### Sebastian Blohm (node 5818)

- Predicted group: **Knowledge Management**
- Model confidence: **1.000**
- Ground-truth match: **True**
- Edges above importance threshold 0.5 (within 2-hop receptive field): **3**
- Table and figure show the top 10 edges by importance

For Sebastian Blohm, the model predicts the research group Knowledge Management. The explanation says this decision is mainly supported by graph connections of type inverse of works At Project, inverse of fax, carries Out. The strongest single piece of evidence is the connection 'X-Media - X-Media' -- inverse of works At Project --> 'Sebastian Blohm' with importance 0.89. In simple terms, the model is looking at this person's surrounding publications, projects, topics, and related entities, then using the most influential links around that person to justify the group prediction.

| Rank | Source | Relation | Target | Importance |
|---:|---|---|---|---:|
| 1 | X-Media - X-Media | worksAtProject_inverse | Sebastian Blohm | 0.887 |
| 2 | +49 (721) 608 6580 | fax_inverse | Sebastian Blohm | 0.857 |
| 3 | Knowledge Management | carriesOut | X-Media - X-Media | 0.844 |
| 4 | +49 (721) 608 7363 | phone_inverse | Sebastian Blohm | 0.297 |
| 5 | EU-IST | financedBy_inverse | X-Media - X-Media | 0.216 |
| 6 | Project | type_inverse | X-Media - X-Media | 0.172 |
| 7 | Person | type_inverse | Sebastian Blohm | 0.166 |
| 8 | U1p2l3o4a5d2131 | photo_inverse | Sebastian Blohm | 0.159 |
| 9 | Information Retrieval | isWorkedOnBy | Sebastian Blohm | 0.158 |
| 10 | Knowledge Management | carriedOutBy_inverse | X-Media - X-Media | 0.157 |

### Peter Bungert (node 5806)

- Predicted group: **Efficient Algorithms**
- Model confidence: **1.000**
- Ground-truth match: **True**
- Edges above importance threshold 0.5 (within 2-hop receptive field): **3**
- Table and figure show the top 10 edges by importance

For Peter Bungert, the model predicts the research group Efficient Algorithms. The explanation says this decision is mainly supported by graph connections of type inverse of works At Project, inverse of fax, carries Out. The strongest single piece of evidence is the connection 'OptRek - Optimierung auf rekonfigurierbaren Rechensystemen' -- inverse of works At Project --> 'Peter Bungert' with importance 0.86. In simple terms, the model is looking at this person's surrounding publications, projects, topics, and related entities, then using the most influential links around that person to justify the group prediction.

| Rank | Source | Relation | Target | Importance |
|---:|---|---|---|---:|
| 1 | OptRek - Optimierung auf rekonfigurierbaren Rechensystemen | worksAtProject_inverse | Peter Bungert | 0.862 |
| 2 | +49 (721) 693717 | fax_inverse | Peter Bungert | 0.836 |
| 3 | Efficient Algorithms | carriesOut | OptRek - Optimierung auf rekonfigurierbaren Rechensystemen | 0.557 |
| 4 | Efficient Algorithms | carriedOutBy_inverse | OptRek - Optimierung auf rekonfigurierbaren Rechensystemen | 0.410 |
| 5 | Peter Bungert | name_inverse | Peter Bungert | 0.167 |
| 6 | U1p2l3o4a5d2119 | photo_inverse | Peter Bungert | 0.157 |
| 7 | Project | type_inverse | OptRek - Optimierung auf rekonfigurierbaren Rechensystemen | 0.153 |
| 8 | Bernd Scheuermann | worksAtProject | OptRek - Optimierung auf rekonfigurierbaren Rechensystemen | 0.145 |
| 9 | Maarten Menken | type | Person | 0.145 |
| 10 | Roland Schätzle | type | Person | 0.144 |

### Saartje Brockmans (node 5772)

- Predicted group: **Knowledge Management**
- Model confidence: **1.000**
- Ground-truth match: **True**
- Edges above importance threshold 0.5 (within 2-hop receptive field): **6**
- Table and figure show the top 10 edges by importance

For Saartje Brockmans, the model predicts the research group Knowledge Management. The explanation says this decision is mainly supported by graph connections of type photo, inverse of photo, inverse of fax. The strongest single piece of evidence is the connection 'Saartje Brockmans' -- photo --> 'U1p2l3o4a5d2077' with importance 0.90. In simple terms, the model is looking at this person's surrounding publications, projects, topics, and related entities, then using the most influential links around that person to justify the group prediction.

| Rank | Source | Relation | Target | Importance |
|---:|---|---|---|---:|
| 1 | Saartje Brockmans | photo | U1p2l3o4a5d2077 | 0.898 |
| 2 | U1p2l3o4a5d2077 | photo_inverse | Saartje Brockmans | 0.849 |
| 3 | +49 (721) 608 6580 | fax_inverse | Saartje Brockmans | 0.843 |
| 4 | knowledge representation and reasoning | isWorkedOnBy | Saartje Brockmans | 0.825 |
| 5 | Graduiertenkolleg IME - Informationswirtschaft und Market Engineering | worksAtProject_inverse | Saartje Brockmans | 0.709 |
| 6 | Knowledge Management | carriesOut | Graduiertenkolleg IME - Informationswirtschaft und Market Engineering | 0.694 |
| 7 | eOrg - Forschungsschwerpunkt eOrganisation | worksAtProject_inverse | Saartje Brockmans | 0.432 |
| 8 | Person | type_inverse | Saartje Brockmans | 0.266 |
| 9 | Saartje Brockmans | name | Saartje Brockmans | 0.242 |
| 10 | Research Topic | type_inverse | knowledge representation and reasoning | 0.190 |
