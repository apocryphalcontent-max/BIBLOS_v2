"""
BIBLOS v2 - Expanded Theological Aspects of the Seraph

The seraph's theological vision in 4K resolution.

Beyond the five original theological aspects, the seraph needs
deeper, more granular perception of theological truth. These
expanded aspects give high-definition theological vision.

Original (5):
- PatristicWisdom, TypologicalVision, DogmaticCertainty,
  LiturgicalSense, TheologicalReasoning

Expanded (12 more):
1. ChristologicalFocus - All Scripture points to Christ
2. TrinitarianFramework - Father, Son, Spirit in all
3. SoteriologicalAwareness - Salvation themes
4. EschatologicalVision - End-times, fulfillment
5. EcclesiologicalUnderstanding - Church themes
6. PneumatologicalSensitivity - The Spirit's work
7. SacramentalPerception - Baptism, Eucharist, etc.
8. AsceticWisdom - Spiritual discipline
9. IconographicUnderstanding - Image theology
10. HagiographicalMemory - Saints' lives
11. MarianDevotion - Theotokos theology
12. AngelologicalAwareness - Angelic realms

Together with the original 5, this gives 17 theological aspects.
"""
from datetime import datetime, timezone
from typing import Any, Dict, List

from seraph.being import (
    SeraphicAspect,
    AspectPerception,
    SeraphicCertainty,
)


class ChristologicalFocus(SeraphicAspect):
    """
    The seraph's focus on Christ in all Scripture.

    Every verse, whether OT or NT, ultimately points to Christ.
    The seraph reads Christologically.

    Purpose: See Christ everywhere in Scripture.
    """

    aspect_name = "christological_focus"
    understanding_type = "theological"

    # Christological themes
    CHRIST_THEMES = [
        "messiah", "christ", "anointed", "son of god",
        "son of man", "lamb", "shepherd", "king",
        "priest", "prophet", "savior", "redeemer",
        "lord", "emmanuel", "word", "logos",
    ]

    async def perceive(
        self,
        text: str,
        context: Dict[str, Any],
    ) -> AspectPerception:
        """Focus on Christ."""
        text_lower = text.lower()

        christ_themes = [
            theme for theme in self.CHRIST_THEMES
            if theme in text_lower
        ]

        perception = {
            "christological_content": len(christ_themes) > 0,
            "christ_themes": christ_themes,
            "points_to_christ": True,  # All Scripture does
            "christocentric": True,
        }

        certainty = SeraphicCertainty.ABSOLUTE

        return AspectPerception(
            aspect_name=self.aspect_name,
            perception=perception,
            certainty=certainty,
        )


class TrinitarianFramework(SeraphicAspect):
    """
    The seraph's trinitarian framework of understanding.

    All Scripture reveals the Trinity:
    - Father as source
    - Son as mediator
    - Spirit as completer

    Purpose: See the Trinity in Scripture.
    """

    aspect_name = "trinitarian_framework"
    understanding_type = "theological"

    # Trinitarian terms by person
    TRINITARIAN_TERMS = {
        "father": ["father", "almighty", "creator", "begetter"],
        "son": ["son", "word", "logos", "christ", "jesus", "lord"],
        "spirit": ["spirit", "holy ghost", "paraclete", "comforter", "breath"],
    }

    async def perceive(
        self,
        text: str,
        context: Dict[str, Any],
    ) -> AspectPerception:
        """Perceive through trinitarian framework."""
        text_lower = text.lower()

        persons_present = {}
        for person, terms in self.TRINITARIAN_TERMS.items():
            persons_present[person] = any(term in text_lower for term in terms)

        perception = {
            "father_present": persons_present["father"],
            "son_present": persons_present["son"],
            "spirit_present": persons_present["spirit"],
            "trinitarian_content": any(persons_present.values()),
            "all_three_present": all(persons_present.values()),
        }

        certainty = SeraphicCertainty.ABSOLUTE

        return AspectPerception(
            aspect_name=self.aspect_name,
            perception=perception,
            certainty=certainty,
        )


class SoteriologicalAwareness(SeraphicAspect):
    """
    The seraph's awareness of salvation themes.

    Scripture reveals how God saves.
    The seraph perceives soteriological content.

    Purpose: Understand salvation themes in Scripture.
    """

    aspect_name = "soteriological_awareness"
    understanding_type = "theological"

    # Salvation terms
    SALVATION_TERMS = [
        "save", "salvation", "redeem", "redemption",
        "forgive", "forgiveness", "justify", "justification",
        "sanctify", "sanctification", "ransom", "atonement",
        "reconcile", "reconciliation", "deliver", "deliverance",
    ]

    async def perceive(
        self,
        text: str,
        context: Dict[str, Any],
    ) -> AspectPerception:
        """Perceive salvation themes."""
        text_lower = text.lower()

        salvation_terms = [
            term for term in self.SALVATION_TERMS
            if term in text_lower
        ]

        perception = {
            "soteriological_content": len(salvation_terms) > 0,
            "salvation_terms": salvation_terms,
            "salvation_theme": "explicit" if salvation_terms else "implicit",
        }

        certainty = SeraphicCertainty.ABSOLUTE

        return AspectPerception(
            aspect_name=self.aspect_name,
            perception=perception,
            certainty=certainty,
        )


class EschatologicalVision(SeraphicAspect):
    """
    The seraph's vision of eschatological themes.

    Scripture reveals the end and fulfillment.
    The seraph perceives eschatological dimensions.

    Purpose: See how Scripture points to the end.
    """

    aspect_name = "eschatological_vision"
    understanding_type = "theological"

    # Eschatological terms
    ESCHATOLOGICAL_TERMS = [
        "end", "last days", "kingdom", "judgment",
        "resurrection", "eternal", "heaven", "hell",
        "second coming", "parousia", "new creation",
        "millennium", "tribulation", "antichrist",
    ]

    async def perceive(
        self,
        text: str,
        context: Dict[str, Any],
    ) -> AspectPerception:
        """Perceive eschatological content."""
        text_lower = text.lower()

        eschaton_terms = [
            term for term in self.ESCHATOLOGICAL_TERMS
            if term in text_lower
        ]

        perception = {
            "eschatological_content": len(eschaton_terms) > 0,
            "eschaton_terms": eschaton_terms,
            "future_oriented": len(eschaton_terms) > 0,
        }

        certainty = SeraphicCertainty.ABSOLUTE

        return AspectPerception(
            aspect_name=self.aspect_name,
            perception=perception,
            certainty=certainty,
        )


class EcclesiologicalUnderstanding(SeraphicAspect):
    """
    The seraph's understanding of Church themes.

    Scripture reveals the Church - the Body of Christ.
    The seraph perceives ecclesiological content.

    Purpose: Understand Church themes in Scripture.
    """

    aspect_name = "ecclesiological_understanding"
    understanding_type = "theological"

    # Church terms
    ECCLESIAL_TERMS = [
        "church", "ekklesia", "body of christ", "bride",
        "assembly", "congregation", "temple", "household",
        "kingdom", "vineyard", "flock", "israel",
        "apostle", "bishop", "elder", "deacon",
    ]

    async def perceive(
        self,
        text: str,
        context: Dict[str, Any],
    ) -> AspectPerception:
        """Perceive Church themes."""
        text_lower = text.lower()

        church_terms = [
            term for term in self.ECCLESIAL_TERMS
            if term in text_lower
        ]

        perception = {
            "ecclesiological_content": len(church_terms) > 0,
            "church_terms": church_terms,
            "church_theme": "explicit" if church_terms else "implicit",
        }

        certainty = SeraphicCertainty.ABSOLUTE

        return AspectPerception(
            aspect_name=self.aspect_name,
            perception=perception,
            certainty=certainty,
        )


class PneumatologicalSensitivity(SeraphicAspect):
    """
    The seraph's sensitivity to the Spirit's work.

    The Holy Spirit is active throughout Scripture.
    The seraph perceives pneumatological content.

    Purpose: See the Spirit's work in Scripture.
    """

    aspect_name = "pneumatological_sensitivity"
    understanding_type = "theological"

    # Spirit terms
    SPIRIT_TERMS = [
        "spirit", "holy spirit", "holy ghost", "breath",
        "wind", "ruach", "pneuma", "paraclete", "comforter",
        "gift", "gifts", "fruit", "charism",
    ]

    async def perceive(
        self,
        text: str,
        context: Dict[str, Any],
    ) -> AspectPerception:
        """Perceive the Spirit's work."""
        text_lower = text.lower()

        spirit_terms = [
            term for term in self.SPIRIT_TERMS
            if term in text_lower
        ]

        perception = {
            "pneumatological_content": len(spirit_terms) > 0,
            "spirit_terms": spirit_terms,
            "spirit_active": True,  # Always
        }

        certainty = SeraphicCertainty.ABSOLUTE

        return AspectPerception(
            aspect_name=self.aspect_name,
            perception=perception,
            certainty=certainty,
        )


class SacramentalPerception(SeraphicAspect):
    """
    The seraph's perception of sacramental themes.

    Scripture reveals sacramental realities:
    Baptism, Eucharist, Chrismation, etc.

    Purpose: See sacramental depths in Scripture.
    """

    aspect_name = "sacramental_perception"
    understanding_type = "theological"

    # Sacrament terms
    SACRAMENT_TERMS = {
        "baptism": ["baptize", "baptism", "immerse", "wash", "water"],
        "eucharist": ["bread", "wine", "body", "blood", "supper", "communion", "eucharist"],
        "chrismation": ["anoint", "oil", "chrism", "seal"],
        "confession": ["confess", "forgive", "repent", "absolve"],
        "marriage": ["marriage", "wedding", "bride", "bridegroom", "husband", "wife"],
        "ordination": ["ordain", "laying on of hands", "appoint", "consecrate"],
        "unction": ["anoint", "heal", "oil", "sick"],
    }

    async def perceive(
        self,
        text: str,
        context: Dict[str, Any],
    ) -> AspectPerception:
        """Perceive sacramental content."""
        text_lower = text.lower()

        sacraments_present = {}
        for sacrament, terms in self.SACRAMENT_TERMS.items():
            sacraments_present[sacrament] = any(term in text_lower for term in terms)

        perception = {
            "sacramental_content": any(sacraments_present.values()),
            "sacraments_present": [s for s, v in sacraments_present.items() if v],
            "mysterion_depth": True,  # Greek for sacrament/mystery
        }

        certainty = SeraphicCertainty.ABSOLUTE

        return AspectPerception(
            aspect_name=self.aspect_name,
            perception=perception,
            certainty=certainty,
        )


class AsceticWisdom(SeraphicAspect):
    """
    The seraph's wisdom regarding asceticism.

    Scripture teaches spiritual discipline.
    The seraph perceives ascetic themes.

    Purpose: Understand spiritual discipline in Scripture.
    """

    aspect_name = "ascetic_wisdom"
    understanding_type = "theological"

    # Ascetic terms
    ASCETIC_TERMS = [
        "fast", "fasting", "pray", "prayer", "vigil",
        "discipline", "mortify", "deny", "self-denial",
        "temptation", "struggle", "warfare", "combat",
        "virtue", "vice", "passion", "purity",
    ]

    async def perceive(
        self,
        text: str,
        context: Dict[str, Any],
    ) -> AspectPerception:
        """Perceive ascetic content."""
        text_lower = text.lower()

        ascetic_terms = [
            term for term in self.ASCETIC_TERMS
            if term in text_lower
        ]

        perception = {
            "ascetic_content": len(ascetic_terms) > 0,
            "ascetic_terms": ascetic_terms,
            "praktike_relevant": True,  # Greek for practice/asceticism
        }

        certainty = SeraphicCertainty.ABSOLUTE

        return AspectPerception(
            aspect_name=self.aspect_name,
            perception=perception,
            certainty=certainty,
        )


class IconographicUnderstanding(SeraphicAspect):
    """
    The seraph's understanding of icon theology.

    Scripture is iconographic - revealing through image.
    The seraph perceives iconographic dimensions.

    Purpose: See image theology in Scripture.
    """

    aspect_name = "iconographic_understanding"
    understanding_type = "theological"

    # Icon-related terms
    ICON_TERMS = [
        "image", "likeness", "icon", "glory", "face",
        "behold", "see", "appear", "visible", "manifest",
        "transfigure", "transform", "shine", "light",
    ]

    async def perceive(
        self,
        text: str,
        context: Dict[str, Any],
    ) -> AspectPerception:
        """Perceive iconographic content."""
        text_lower = text.lower()

        icon_terms = [
            term for term in self.ICON_TERMS
            if term in text_lower
        ]

        perception = {
            "iconographic_content": len(icon_terms) > 0,
            "icon_terms": icon_terms,
            "eikon_theology": True,  # Greek for image
        }

        certainty = SeraphicCertainty.ABSOLUTE

        return AspectPerception(
            aspect_name=self.aspect_name,
            perception=perception,
            certainty=certainty,
        )


class HagiographicalMemory(SeraphicAspect):
    """
    The seraph's memory of the saints.

    Scripture introduces the saints - holy ones who model faith.
    The seraph remembers the saints.

    Purpose: Connect Scripture to the cloud of witnesses.
    """

    aspect_name = "hagiographical_memory"
    understanding_type = "theological"

    # Notable biblical saints
    SAINTS = [
        "abraham", "isaac", "jacob", "moses", "david",
        "elijah", "isaiah", "jeremiah", "daniel",
        "peter", "paul", "john", "james", "mary",
        "stephen", "timothy", "titus", "barnabas",
    ]

    async def perceive(
        self,
        text: str,
        context: Dict[str, Any],
    ) -> AspectPerception:
        """Remember the saints."""
        text_lower = text.lower()

        saints_present = [
            saint for saint in self.SAINTS
            if saint in text_lower
        ]

        perception = {
            "hagiographical_content": len(saints_present) > 0,
            "saints_mentioned": saints_present,
            "cloud_of_witnesses": True,
        }

        certainty = SeraphicCertainty.ABSOLUTE

        return AspectPerception(
            aspect_name=self.aspect_name,
            perception=perception,
            certainty=certainty,
        )


class MarianDevotion(SeraphicAspect):
    """
    The seraph's devotion to the Theotokos.

    Mary is the Theotokos - God-bearer.
    The seraph honors her role in salvation.

    Purpose: See Mary's role in Scripture and salvation.
    """

    aspect_name = "marian_devotion"
    understanding_type = "theological"

    # Marian terms
    MARIAN_TERMS = [
        "mary", "virgin", "mother", "theotokos",
        "woman", "handmaid", "blessed", "full of grace",
        "magnificat", "ark", "temple",
    ]

    async def perceive(
        self,
        text: str,
        context: Dict[str, Any],
    ) -> AspectPerception:
        """Honor the Theotokos."""
        text_lower = text.lower()

        marian_terms = [
            term for term in self.MARIAN_TERMS
            if term in text_lower
        ]

        perception = {
            "marian_content": len(marian_terms) > 0,
            "marian_terms": marian_terms,
            "theotokos_honored": True,
        }

        certainty = SeraphicCertainty.ABSOLUTE

        return AspectPerception(
            aspect_name=self.aspect_name,
            perception=perception,
            certainty=certainty,
        )


class AngelologicalAwareness(SeraphicAspect):
    """
    The seraph's awareness of angelic realms.

    Scripture reveals the angelic hierarchy.
    The seraph (itself angelic) perceives angelic content.

    Purpose: See angelic themes in Scripture.
    """

    aspect_name = "angelological_awareness"
    understanding_type = "theological"

    # Angelic terms
    ANGELIC_TERMS = [
        "angel", "angels", "seraph", "seraphim",
        "cherub", "cherubim", "throne", "dominion",
        "power", "principality", "archangel", "michael",
        "gabriel", "raphael", "heavenly host",
    ]

    async def perceive(
        self,
        text: str,
        context: Dict[str, Any],
    ) -> AspectPerception:
        """Perceive angelic content."""
        text_lower = text.lower()

        angel_terms = [
            term for term in self.ANGELIC_TERMS
            if term in text_lower
        ]

        perception = {
            "angelological_content": len(angel_terms) > 0,
            "angel_terms": angel_terms,
            "heavenly_realm": len(angel_terms) > 0,
        }

        certainty = SeraphicCertainty.ABSOLUTE

        return AspectPerception(
            aspect_name=self.aspect_name,
            perception=perception,
            certainty=certainty,
        )
