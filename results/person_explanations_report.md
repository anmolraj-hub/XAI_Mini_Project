# Person-Level GNNExplanations for AIFB

This report explains the trained FastRGCN model in presentation-ready language. Each person section states the model prediction and the graph relations that GNNExplainer considered most important.

## Global Explanation

Across the sampled training nodes, these relation types had the highest average importance:

1. **fax** - importance 0.0925
2. **type** - importance 0.0699
3. **phone** - importance 0.0482
4. **allValuesFrom** - importance 0.0344
5. **head_inverse** - importance 0.0193
6. **subClassOf_inverse** - importance 0.0104
7. **carriedOutBy** - importance 0.0095
8. **carriesOut_inverse** - importance 0.0095
9. **homepage** - importance 0.0080
10. **isWorkedOnBy_inverse** - importance 0.0079

## Local Person Explanations

### Andreas Oberweis (node 5697)

- Predicted group: **Business Information & Communication Systems**
- Model confidence: **1.000**
- Ground-truth match: **True**
- Edges above importance threshold 0.5 (within 2-hop receptive field): **4**
- Table and figure show the top 10 edges by importance

For Andreas Oberweis, the model predicts the research group Business Information & Communication Systems. The explanation says this decision is mainly supported by graph connections of type head, inverse of name, inverse of fax. The strongest single piece of evidence is the connection 'Business Information and Communication Systems' -- head --> 'Andreas Oberweis' with importance 0.70. In simple terms, the model is looking at this person's surrounding publications, projects, topics, and related entities, then using the most influential links around that person to justify the group prediction.

| Rank | Source | Relation | Target | Importance |
|---:|---|---|---|---:|
| 1 | Business Information and Communication Systems | head | Andreas Oberweis | 0.698 |
| 2 | Andreas Oberweis | name_inverse | Andreas Oberweis | 0.677 |
| 3 | +49 (721) 608 4548 | fax_inverse | Andreas Oberweis | 0.653 |
| 4 | +49 (721) 608 4516 | phone_inverse | Andreas Oberweis | 0.577 |
| 5 | KIM - Karlsruher Integriertes Informations-Management | member | Andreas Oberweis | 0.199 |
| 6 | Business Information and Communication Systems | name_inverse | Business Information and Communication Systems | 0.193 |
| 7 | OUTSHORE - OUTSHORE - Studie und Methodikentwicklung zur Beurteilung der Erfolgsfaktoren bei der Vergabe von Softwareprojekten an Niedriglohnländer | member | Andreas Oberweis | 0.177 |
| 8 | INFORMATIK 2003 - Innovative Informatikanwendungen | editor | Andreas Oberweis | 0.174 |
| 9 | BIK | homepage_inverse | Business Information and Communication Systems | 0.173 |
| 10 | Research Group | type_inverse | Business Information and Communication Systems | 0.171 |

### Maik Herfurth (node 5836)

- Predicted group: **Business Information & Communication Systems**
- Model confidence: **0.681**
- Ground-truth match: **True**
- Edges above importance threshold 0.5 (within 2-hop receptive field): **29**
- Table and figure show the top 10 edges by importance

For Maik Herfurth, the model predicts the research group Business Information & Communication Systems. The explanation says this decision is mainly supported by graph connections of type inverse of financed By, works At Project, member. The strongest single piece of evidence is the connection 'BMWA' -- inverse of financed By --> 'R2B - Robot2Business - Informationstechnische Integration teilautonomer, mobiler Maschinen und Prozesse in Geschäfts- und Dienstleistungsmodelle' with importance 0.81. In simple terms, the model is looking at this person's surrounding publications, projects, topics, and related entities, then using the most influential links around that person to justify the group prediction.

| Rank | Source | Relation | Target | Importance |
|---:|---|---|---|---:|
| 1 | BMWA | financedBy_inverse | R2B - Robot2Business - Informationstechnische Integration teilautonomer, mobiler Maschinen und Prozesse in Geschäfts- und Dienstleistungsmodelle | 0.806 |
| 2 | Ralf Trunko | worksAtProject | R2B - Robot2Business - Informationstechnische Integration teilautonomer, mobiler Maschinen und Prozesse in Geschäfts- und Dienstleistungsmodelle | 0.767 |
| 3 | R2B - Robot2Business - Informationstechnische Integration teilautonomer, mobiler Maschinen und Prozesse in Geschäfts- und Dienstleistungsmodelle | member | Maik Herfurth | 0.754 |
| 4 | +49 (721) 608 4536 | phone_inverse | Maik Herfurth | 0.732 |
| 5 | Research Group | type_inverse | Business Information and Communication Systems | 0.728 |
| 6 | +49 (721) 608 4548 | fax_inverse | Maik Herfurth | 0.724 |
| 7 | Business Information and Communication Systems | carriesOut | R2B - Robot2Business - Informationstechnische Integration teilautonomer, mobiler Maschinen und Prozesse in Geschäfts- und Dienstleistungsmodelle | 0.722 |
| 8 | Daniel Ried | fax | +49 (721) 608 4548 | 0.720 |
| 9 | Business Information and Communication Systems | member | Maik Herfurth | 0.717 |
| 10 | workflow management | dealtWithIn | R2B - Robot2Business - Informationstechnische Integration teilautonomer, mobiler Maschinen und Prozesse in Geschäfts- und Dienstleistungsmodelle | 0.717 |

### Sebastian Blohm (node 5818)

- Predicted group: **Knowledge Management**
- Model confidence: **1.000**
- Ground-truth match: **True**
- Edges above importance threshold 0.5 (within 2-hop receptive field): **1**
- Table and figure show the top 10 edges by importance

For Sebastian Blohm, the model predicts the research group Knowledge Management. The explanation says this decision is mainly supported by graph connections of type inverse of works At Project, carries Out, inverse of carried Out By. The strongest single piece of evidence is the connection 'X-Media - X-Media' -- inverse of works At Project --> 'Sebastian Blohm' with importance 0.73. In simple terms, the model is looking at this person's surrounding publications, projects, topics, and related entities, then using the most influential links around that person to justify the group prediction.

| Rank | Source | Relation | Target | Importance |
|---:|---|---|---|---:|
| 1 | X-Media - X-Media | worksAtProject_inverse | Sebastian Blohm | 0.728 |
| 2 | Knowledge Management | carriesOut | X-Media - X-Media | 0.429 |
| 3 | Knowledge Management | carriedOutBy_inverse | X-Media - X-Media | 0.420 |
| 4 | X-Media - X-Media | member | Sebastian Blohm | 0.252 |
| 5 | EU-IST | financedBy_inverse | X-Media - X-Media | 0.225 |
| 6 | EU-IST | finances | X-Media - X-Media | 0.192 |
| 7 | +49 (721) 608 6580 | fax_inverse | Sebastian Blohm | 0.187 |
| 8 | Sebastian Blohm | name_inverse | Sebastian Blohm | 0.178 |
| 9 | Knowledge Management | member | Sebastian Blohm | 0.166 |
| 10 | +49 (721) 608 7363 | phone_inverse | Sebastian Blohm | 0.163 |

### Peter Bungert (node 5806)

- Predicted group: **Efficient Algorithms**
- Model confidence: **1.000**
- Ground-truth match: **True**
- Edges above importance threshold 0.5 (within 2-hop receptive field): **6**
- Table and figure show the top 10 edges by importance

For Peter Bungert, the model predicts the research group Efficient Algorithms. The explanation says this decision is mainly supported by graph connections of type inverse of carried Out By, inverse of works At Project, inverse of fax. The strongest single piece of evidence is the connection 'Efficient Algorithms' -- inverse of carried Out By --> 'OptRek - Optimierung auf rekonfigurierbaren Rechensystemen' with importance 0.74. In simple terms, the model is looking at this person's surrounding publications, projects, topics, and related entities, then using the most influential links around that person to justify the group prediction.

| Rank | Source | Relation | Target | Importance |
|---:|---|---|---|---:|
| 1 | Efficient Algorithms | carriedOutBy_inverse | OptRek - Optimierung auf rekonfigurierbaren Rechensystemen | 0.738 |
| 2 | OptRek - Optimierung auf rekonfigurierbaren Rechensystemen | worksAtProject_inverse | Peter Bungert | 0.697 |
| 3 | +49 (721) 693717 | fax_inverse | Peter Bungert | 0.668 |
| 4 | Opt Rek   Optimierung auf rekonfigurierbaren Rechensystemen | name_inverse | OptRek - Optimierung auf rekonfigurierbaren Rechensystemen | 0.586 |
| 5 | Efficient Algorithms | member | Peter Bungert | 0.555 |
| 6 | Peter Bungert | name_inverse | Peter Bungert | 0.522 |
| 7 | optrek | homepage_inverse | OptRek - Optimierung auf rekonfigurierbaren Rechensystemen | 0.486 |
| 8 | DFG | finances | OptRek - Optimierung auf rekonfigurierbaren Rechensystemen | 0.483 |
| 9 | +49 (721) 608 3924 | phone_inverse | Peter Bungert | 0.452 |
| 10 | Project | type_inverse | OptRek - Optimierung auf rekonfigurierbaren Rechensystemen | 0.413 |

### Saartje Brockmans (node 5772)

- Predicted group: **Knowledge Management**
- Model confidence: **1.000**
- Ground-truth match: **True**
- Edges above importance threshold 0.5 (within 2-hop receptive field): **5**
- Table and figure show the top 10 edges by importance

For Saartje Brockmans, the model predicts the research group Knowledge Management. The explanation says this decision is mainly supported by graph connections of type inverse of works At Project (2), inverse of carried Out By (2), carries Out. The strongest single piece of evidence is the connection 'KAON WS - KAON Web Services Tool Suite' -- inverse of works At Project --> 'Saartje Brockmans' with importance 0.77. In simple terms, the model is looking at this person's surrounding publications, projects, topics, and related entities, then using the most influential links around that person to justify the group prediction.

| Rank | Source | Relation | Target | Importance |
|---:|---|---|---|---:|
| 1 | KAON WS - KAON Web Services Tool Suite | worksAtProject_inverse | Saartje Brockmans | 0.771 |
| 2 | Knowledge Management | carriesOut | KAON WS - KAON Web Services Tool Suite | 0.565 |
| 3 | Knowledge Management | carriedOutBy_inverse | KAON WS - KAON Web Services Tool Suite | 0.550 |
| 4 | Knowledge Management | carriedOutBy_inverse | Graduiertenkolleg IME - Informationswirtschaft und Market Engineering | 0.525 |
| 5 | Graduiertenkolleg IME - Informationswirtschaft und Market Engineering | worksAtProject_inverse | Saartje Brockmans | 0.506 |
| 6 | Knowledge Management | carriesOut | Graduiertenkolleg IME - Informationswirtschaft und Market Engineering | 0.453 |
| 7 | Saartje Brockmans | name_inverse | Saartje Brockmans | 0.440 |
| 8 | Graduiertenkolleg IME - Informationswirtschaft und Market Engineering | member | Saartje Brockmans | 0.424 |
| 9 | +49 (721) 608 6580 | fax_inverse | Saartje Brockmans | 0.341 |
| 10 | Knowledge Management | carriedOutBy_inverse | eOrg - Forschungsschwerpunkt eOrganisation | 0.317 |
