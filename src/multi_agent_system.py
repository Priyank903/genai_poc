"""
Multi-Agent System for Post-Discharge Patient Care
Implements Receptionist Agent and Clinical AI Agent with proper workflows
"""

import json
import logging
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from improved_rag_system import ImprovedRAGSystem
from web_search import WebSearchTool

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentType(Enum):
    RECEPTIONIST = "receptionist"
    CLINICAL = "clinical"

@dataclass
class PatientData:
    """Patient discharge report data structure"""
    patient_name: str
    discharge_date: str
    primary_diagnosis: str
    medications: List[str]
    dietary_restrictions: str
    follow_up: str
    warning_signs: str
    discharge_instructions: str

@dataclass
class Interaction:
    """Interaction log entry"""
    timestamp: str
    agent_type: AgentType
    user_input: str
    agent_response: str
    patient_name: Optional[str] = None
    action_taken: Optional[str] = None
    metadata: Optional[Dict] = None

class PatientDatabase:
    """Patient data retrieval tool"""
    
    def __init__(self, patients_file: str = None):
        if patients_file is None:
            # Try different possible paths
            possible_paths = [
                "../data/patients.json",
                "data/patients.json", 
                os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "patients.json")
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    patients_file = path
                    break
            else:
                patients_file = "../data/patients.json"  # Default fallback
        
        self.patients_file = patients_file
        self.patients_data = self._load_patients()
        logger.info(f"Loaded {len(self.patients_data)} patient records")
    
    def _load_patients(self) -> List[PatientData]:
        """Load patient data from JSON file"""
        try:
            with open(self.patients_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            patients = []
            for patient_dict in data:
                patient = PatientData(**patient_dict)
                patients.append(patient)
            
            return patients
        except Exception as e:
            logger.error(f"Error loading patients: {str(e)}")
            return []
    
    def find_patient_by_name(self, name: str) -> Optional[PatientData]:
        """
        Find patient by name (case-insensitive, partial match)
        
        Args:
            name: Patient name to search for
            
        Returns:
            PatientData if found, None otherwise
        """
        name_lower = name.lower().strip()
        
        # Exact match first
        for patient in self.patients_data:
            if patient.patient_name.lower() == name_lower:
                logger.info(f"Found exact match for patient: {patient.patient_name}")
                return patient
        
        # Partial match
        matches = []
        for patient in self.patients_data:
            if name_lower in patient.patient_name.lower():
                matches.append(patient)
        
        if len(matches) == 1:
            logger.info(f"Found partial match for patient: {matches[0].patient_name}")
            return matches[0]
        elif len(matches) > 1:
            logger.warning(f"Multiple patients found for '{name}': {[p.patient_name for p in matches]}")
            return matches[0]  # Return first match
        
        logger.warning(f"No patient found for name: {name}")
        return None
    
    def get_all_patient_names(self) -> List[str]:
        """Get list of all patient names"""
        return [patient.patient_name for patient in self.patients_data]

class InteractionLogger:
    """Comprehensive logging system for agent interactions"""
    
    def __init__(self, log_file: str = "../logs/agent_interactions.json"):
        self.log_file = log_file
        self.interactions: List[Interaction] = []
        self._ensure_log_file()
    
    def _ensure_log_file(self):
        """Ensure log file and directory exist"""
        import os
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                json.dump([], f)
    
    def log_interaction(self, interaction: Interaction):
        """Log an interaction"""
        self.interactions.append(interaction)
        
        # Simple logging - just keep in memory for now
        try:
            # Add new interaction
            interaction_dict = {
                "timestamp": interaction.timestamp,
                "agent_type": str(interaction.agent_type),
                "user_input": interaction.user_input,
                "agent_response": interaction.agent_response[:200] + "..." if len(interaction.agent_response) > 200 else interaction.agent_response,
                "patient_name": interaction.patient_name,
                "action_taken": interaction.action_taken
            }
            logger.info(f"Interaction logged: {interaction_dict}")
                
        except Exception as e:
            logger.error(f"Error logging interaction: {str(e)}")

class ReceptionistAgent:
    """
    Receptionist Agent handles initial patient interaction and data retrieval
    """
    
    def __init__(self, patient_db: PatientDatabase, interaction_logger: InteractionLogger):
        self.patient_db = patient_db
        self.logger = interaction_logger
        self.current_patient: Optional[PatientData] = None
        self.conversation_state = "greeting"  # greeting, name_collection, patient_found, follow_up
        
        logger.info("Receptionist Agent initialized")
    
    def process_message(self, user_input: str) -> Dict[str, Any]:
        """
        Process user message and return appropriate response
        
        Args:
            user_input: User's message
            
        Returns:
            Response dictionary with message and metadata
        """
        timestamp = datetime.now().isoformat()
        
        if self.conversation_state == "greeting":
            return self._handle_greeting(user_input, timestamp)
        elif self.conversation_state == "name_collection":
            return self._handle_name_collection(user_input, timestamp)
        elif self.conversation_state == "patient_found":
            return self._handle_patient_interaction(user_input, timestamp)
        else:
            return self._handle_general_interaction(user_input, timestamp)
    
    def _handle_greeting(self, user_input: str, timestamp: str) -> Dict[str, Any]:
        """Handle initial greeting"""
        response = "Hello! I'm your post-discharge care assistant. What's your name?"
        
        self.conversation_state = "name_collection"
        
        interaction = Interaction(
            timestamp=timestamp,
            agent_type=AgentType.RECEPTIONIST,
            user_input=user_input,
            agent_response=response,
            action_taken="greeting_provided"
        )
        self.logger.log_interaction(interaction)
        
        return {
            "response": response,
            "agent": "receptionist",
            "state": self.conversation_state,
            "requires_name": True
        }
    
    def _handle_name_collection(self, user_input: str, timestamp: str) -> Dict[str, Any]:
        """Handle name collection and patient lookup"""
        # Extract potential name from input
        name = self._extract_name_from_input(user_input)
        
        if not name:
            response = "I didn't catch your name clearly. Could you please tell me your full name?"
            
            interaction = Interaction(
                timestamp=timestamp,
                agent_type=AgentType.RECEPTIONIST,
                user_input=user_input,
                agent_response=response,
                action_taken="name_clarification_requested"
            )
            self.logger.log_interaction(interaction)
            
            return {
                "response": response,
                "agent": "receptionist",
                "state": self.conversation_state,
                "requires_name": True
            }
        
        # Look up patient
        patient = self.patient_db.find_patient_by_name(name)
        
        if patient:
            self.current_patient = patient
            self.conversation_state = "patient_found"
            
            response = f"Hi {patient.patient_name}! I found your discharge report from {patient.discharge_date} for {patient.primary_diagnosis}. How are you feeling today? Are you following your medication schedule?"
            
            interaction = Interaction(
                timestamp=timestamp,
                agent_type=AgentType.RECEPTIONIST,
                user_input=user_input,
                agent_response=response,
                patient_name=patient.patient_name,
                action_taken="patient_found_and_greeted",
                metadata={"diagnosis": patient.primary_diagnosis, "discharge_date": patient.discharge_date}
            )
            self.logger.log_interaction(interaction)
            
            return {
                "response": response,
                "agent": "receptionist",
                "state": self.conversation_state,
                "patient_found": True,
                "patient_data": patient.__dict__
            }
        else:
            response = f"""I couldn't find a discharge report for "{name}" in our system. This could be because:

1. The name might be spelled differently in our records
2. You might not be in our current database
3. There might be a slight variation in how your name is recorded

Could you try providing your name again, or check if there might be an alternative spelling? You can also contact our main desk at [phone number] for assistance."""
            
            interaction = Interaction(
                timestamp=timestamp,
                agent_type=AgentType.RECEPTIONIST,
                user_input=user_input,
                agent_response=response,
                action_taken="patient_not_found",
                metadata={"searched_name": name}
            )
            self.logger.log_interaction(interaction)
            
            return {
                "response": response,
                "agent": "receptionist",
                "state": "name_collection",
                "patient_found": False,
                "error": "patient_not_found"
            }
    
    def _handle_patient_interaction(self, user_input: str, timestamp: str) -> Dict[str, Any]:
        """Handle interaction with identified patient"""
        # Check if this is a medical question that should be routed to Clinical Agent
        if self._is_medical_question(user_input):
            response = f"""This sounds like a medical question that I should connect you with our Clinical AI Agent for a more detailed answer.

Let me transfer you to our Clinical Agent who has access to comprehensive nephrology reference materials and can provide more specific medical guidance.

Your question: "{user_input}"

**Transferring to Clinical Agent...**"""
            
            interaction = Interaction(
                timestamp=timestamp,
                agent_type=AgentType.RECEPTIONIST,
                user_input=user_input,
                agent_response=response,
                patient_name=self.current_patient.patient_name,
                action_taken="routing_to_clinical_agent",
                metadata={"question_type": "medical"}
            )
            self.logger.log_interaction(interaction)
            
            return {
                "response": response,
                "agent": "receptionist",
                "route_to": "clinical",
                "patient_data": self.current_patient.__dict__,
                "original_question": user_input
            }
        else:
            # Handle non-medical questions with patient data
            return self._handle_discharge_info_question(user_input, timestamp)
    
    def _handle_discharge_info_question(self, user_input: str, timestamp: str) -> Dict[str, Any]:
        """Handle questions about discharge information"""
        patient = self.current_patient
        
        # Simple keyword matching for discharge info
        user_lower = user_input.lower()
        
        if any(word in user_lower for word in ["medication", "medicine", "drug", "pill"]):
            response = f"""Here are your current medications from your discharge report:

**Medications:**
{chr(10).join(f'• {med}' for med in patient.medications)}

**Instructions:** {patient.discharge_instructions}

Please make sure to take these exactly as prescribed. If you have questions about side effects or interactions, I can connect you with our Clinical Agent for more detailed information."""
        
        elif any(word in user_lower for word in ["diet", "food", "eat", "restriction"]):
            response = f"""Here are your dietary restrictions and recommendations:

**Dietary Restrictions:** {patient.dietary_restrictions}

**Discharge Instructions:** {patient.discharge_instructions}

Following these dietary guidelines is important for your recovery. If you need help with meal planning or have questions about specific foods, I can connect you with our Clinical Agent."""
        
        elif any(word in user_lower for word in ["follow", "appointment", "visit", "clinic"]):
            response = f"""Here's your follow-up schedule:

**Follow-up:** {patient.follow_up}

Please make sure to keep these appointments as they're important for monitoring your recovery. If you need help scheduling or have questions about what to expect, let me know!"""
        
        elif any(word in user_lower for word in ["warning", "symptom", "sign", "emergency"]):
            response = f"""This sounds like a medical question about symptoms and warning signs. Let me connect you with our Clinical Agent who can provide comprehensive medical guidance.

**Transferring to Clinical Agent for detailed medical information...**"""
            
            interaction = Interaction(
                timestamp=timestamp,
                agent_type=AgentType.RECEPTIONIST,
                user_input=user_input,
                agent_response=response,
                patient_name=patient.patient_name,
                action_taken="routing_to_clinical_agent_for_warning_signs",
                metadata={"question_type": "warning_signs"}
            )
            self.logger.log_interaction(interaction)
            
            return {
                "response": response,
                "agent": "receptionist",
                "route_to": "clinical",
                "patient_data": patient.__dict__,
                "original_question": user_input
            }
        
        else:
            response = f"""I can help you with information from your discharge report. Here's a summary:

**Diagnosis:** {patient.primary_diagnosis}
**Discharge Date:** {patient.discharge_date}
**Follow-up:** {patient.follow_up}

I can provide more details about:
• Your medications
• Dietary restrictions  
• Warning signs to watch for
• Discharge instructions

What specific information would you like to know more about?"""
        
        interaction = Interaction(
            timestamp=timestamp,
            agent_type=AgentType.RECEPTIONIST,
            user_input=user_input,
            agent_response=response,
            patient_name=patient.patient_name,
            action_taken="provided_discharge_info"
        )
        self.logger.log_interaction(interaction)
        
        return {
            "response": response,
            "agent": "receptionist",
            "state": self.conversation_state
        }
    
    def _handle_general_interaction(self, user_input: str, timestamp: str) -> Dict[str, Any]:
        """Handle general interactions"""
        response = """I'm here to help with your post-discharge care questions. 

If you haven't provided your name yet, please tell me your name so I can look up your discharge information.

If you have medical questions or concerns about symptoms, I can connect you with our Clinical Agent who has access to comprehensive medical references."""
        
        interaction = Interaction(
            timestamp=timestamp,
            agent_type=AgentType.RECEPTIONIST,
            user_input=user_input,
            agent_response=response,
            action_taken="general_assistance_provided"
        )
        self.logger.log_interaction(interaction)
        
        return {
            "response": response,
            "agent": "receptionist",
            "state": "greeting"
        }
    
    def _extract_name_from_input(self, user_input: str) -> str:
        """Extract name from user input"""
        # Simple name extraction - look for capitalized words
        import re
        
        # Remove common greetings and words
        cleaned = re.sub(r'\b(hi|hello|hey|my|name|is|i\'m|im|am)\b', '', user_input.lower())
        
        # Look for capitalized words in original input
        words = user_input.split()
        name_words = []
        
        for word in words:
            # Skip common words but keep capitalized ones that might be names
            if (word[0].isupper() and 
                word.lower() not in ['hi', 'hello', 'hey', 'my', 'name', 'is', 'i', 'am']):
                name_words.append(word)
        
        return ' '.join(name_words) if name_words else user_input.strip()
    
    def _is_medical_question(self, user_input: str) -> bool:
        """Determine if question should be routed to Clinical Agent"""
        medical_keywords = [
            'pain', 'hurt', 'ache', 'swelling', 'swollen', 'blood', 'urine',
            'symptom', 'side effect', 'side effects', 'reaction', 'dizzy', 'nausea', 'vomit',
            'fever', 'infection', 'rash', 'breathing', 'chest', 'heart',
            'kidney', 'dialysis', 'creatinine', 'protein', 'edema',
            'hypertension', 'diabetes', 'medication interaction', 'drug', 'medication',
            'treatment', 'therapy', 'procedure', 'surgery', 'biopsy',
            'indication', 'indications', 'mechanism', 'dose', 'dosage', 'contraindication',
            'contraindications', 'pharmacology', 'tablet', 'capsule', 'pill', 'prescription'
        ]

        medication_phrases = [
            'use of', 'used for', 'purpose of', 'what is the use', 'what is this medicine',
            'what does', 'how does', 'why is', 'benefit of', 'benefits of'
        ]
        
        user_lower = user_input.lower()
        if any(keyword in user_lower for keyword in medical_keywords):
            return True
        if any(phrase in user_lower for phrase in medication_phrases):
            return True
        # Detect potential medication names (words containing typical drug suffixes)
        drug_suffixes = ('zol', 'mab', 'pam', 'pril', 'artan', 'azole', 'dine', 'tide', 'xine', 'xone', 'xane', 'mine', 'mide', 'nib', 'tamine', 'caine', 'oxetine', 'afil', 'sartan', 'gliptin', 'prazole', 'oxetine', 'oxetine', 'topiramate')
        for token in user_lower.split():
            clean = ''.join(ch for ch in token if ch.isalpha())
            if len(clean) >= 6 and clean.endswith(drug_suffixes):
                return True
        return False

class ClinicalAgent:
    """
    Clinical AI Agent handles medical questions using RAG and web search
    """
    
    def __init__(self, rag_system: ImprovedRAGSystem, interaction_logger: InteractionLogger):
        self.rag_system = rag_system
        self.logger = interaction_logger
        self.web_search = WebSearchTool()
        
        logger.info("Clinical Agent initialized")
    
    def process_medical_query(self, query: str, patient_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Process medical query using RAG system with patient context
        
        Args:
            query: Medical question
            patient_data: Patient information for context
            
        Returns:
            Response with medical information and citations
        """
        timestamp = datetime.now().isoformat()
        
        # Add patient context to query if available
        if patient_data:
            contextualized_query = f"""Patient context: {patient_data.get('primary_diagnosis', 'Unknown diagnosis')}
Current medications: {', '.join(patient_data.get('medications', []))}
Question: {query}"""
        else:
            contextualized_query = query
        
        # Use RAG system to get answer
        rag_response = self.rag_system.answer_query(contextualized_query, use_web_fallback=True)
        
        # Format response with medical disclaimers
        formatted_response = self._format_medical_response(rag_response, patient_data)
        
        # Log interaction
        interaction = Interaction(
            timestamp=timestamp,
            agent_type=AgentType.CLINICAL,
            user_input=query,
            agent_response=formatted_response["response"],
            patient_name=patient_data.get('patient_name') if patient_data else None,
            action_taken=f"medical_query_answered_via_{rag_response['source']}",
            metadata={
                "chunks_found": rag_response.get('chunks_found', 0),
                "source": rag_response['source'],
                "has_patient_context": patient_data is not None
            }
        )
        self.logger.log_interaction(interaction)
        
        return formatted_response
    
    def _format_medical_response(self, rag_response: Dict, patient_data: Optional[Dict]) -> Dict[str, Any]:
        """Format RAG response with proper medical disclaimers and citations"""
        
        # Medical disclaimer
        disclaimer = """
**⚠️ Note:** This is an AI assistant for educational purposes only. Always consult healthcare professionals for medical advice."""
        
        # Format main response
        answer = rag_response.get('answer', 'No answer available')
        source = rag_response.get('source', 'unknown')
        
        if source == "rag_with_sarvam":
            # Include detailed citations from RAG
            citations = ""
            if 'relevant_chunks' in rag_response:
                citations = "\n\n**📚 SOURCES FROM NEPHROLOGY REFERENCE MATERIALS:**\n"
                for i, chunk in enumerate(rag_response['relevant_chunks'][:3], 1):
                    citations += f"{i}. **{chunk['source']}** - Page {chunk['page']}\n"
                    citations += f"   Relevance Score: {chunk['similarity_score']:.3f}\n"
                    citations += f"   Text: \"{chunk['text']}\"\n\n"
            
            response_text = f"""**📖 Answer based on RAG (Retrieval-Augmented Generation) from medical references:**

{answer}

{citations}

**🔍 Information Source:** Retrieved from comprehensive nephrology textbook using semantic search

{disclaimer}"""
        
        elif source == "rag_chunks_summary":
            # Sarvam indicated insufficient context; provide synthesized summary from chunks
            citations = ""
            if 'relevant_chunks' in rag_response:
                citations = "\n\n**📚 SOURCES FROM NEPHROLOGY REFERENCE MATERIALS:**\n"
                for i, chunk in enumerate(rag_response['relevant_chunks'][:5], 1):
                    citations += f"{i}. **{chunk['source']}** - Page {chunk['page']}\n"
                    citations += f"   Relevance Score: {chunk['similarity_score']:.3f}\n"
                    citations += f"   Text: \"{chunk['text']}\"\n\n"

            response_text = f"""**📖 Summary based on retrieved nephrology references:**

{answer}

{citations}

**🔍 Information Source:** Directly summarized from nephrology textbook excerpts

{disclaimer}"""

        elif source == "rag_chunks_only":
            # RAG chunks available but Sarvam AI failed
            citations = ""
            if 'relevant_chunks' in rag_response:
                citations = "\n\n**📚 SOURCES FROM NEPHROLOGY REFERENCE MATERIALS:**\n"
                for i, chunk in enumerate(rag_response['relevant_chunks'][:3], 1):
                    citations += f"{i}. **{chunk['source']}** - Page {chunk['page']}\n"
                    citations += f"   Relevance Score: {chunk['similarity_score']:.3f}\n"
                    citations += f"   Text: \"{chunk['text']}\"\n\n"
            
            response_text = f"""**📖 Answer based on RAG chunks (AI processing unavailable):**

{answer}

{citations}

**🔍 Information Source:** Direct excerpts from nephrology textbook (AI summarization failed)

**⚠️ Note:** These are direct excerpts from medical references. AI processing was unavailable for summarization.

{disclaimer}"""
        
        elif source == "web_search_with_rag_context":
            # RAG context was insufficient, used web search with LLM processing
            citations = ""
            if 'relevant_chunks' in rag_response:
                citations = "\n\n**📚 RETRIEVED RAG CHUNKS (INSUFFICIENT FOR ANSWER):**\n"
                for i, chunk in enumerate(rag_response['relevant_chunks'][:3], 1):
                    citations += f"{i}. **{chunk['source']}** - Page {chunk['page']}\n"
                    citations += f"   Relevance Score: {chunk['similarity_score']:.3f}\n"
                    citations += f"   Text: \"{chunk['text']}\"\n\n"
            
            web_citations = ""
            if 'web_search_results' in rag_response:
                web_citations = "\n\n**🌐 WEB SEARCH SOURCES USED:**\n"
                for i, result in enumerate(rag_response['web_search_results'][:5], 1):
                    web_citations += f"{i}. **{result.get('title', 'Unknown')}**\n"
                    web_citations += f"   URL: {result.get('url', 'N/A')}\n"
                    web_citations += f"   Content: \"{result.get('snippet', result.get('content', ''))}\"\n\n"
            
            response_text = f"""**🌐 Answer based on web search (RAG context insufficient):**

{answer}

{citations}

{web_citations}

**🔍 Information Source:** Web search processed by AI (nephrology references were insufficient)

**⚠️ Note:** The retrieved nephrology references did not contain sufficient information to answer this question, so web search was used. Please verify with healthcare professionals.

{disclaimer}"""

        elif source == "web_search_fallback":
            # Web search fallback with LLM processing - ALWAYS show RAG chunks
            citations = ""
            if 'relevant_chunks' in rag_response:
                citations = "\n\n**📚 RETRIEVED RAG CHUNKS:**\n"
                for i, chunk in enumerate(rag_response['relevant_chunks'][:3], 1):
                    citations += f"{i}. **{chunk['source']}** - Page {chunk['page']}\n"
                    citations += f"   Relevance Score: {chunk['similarity_score']:.3f}\n"
                    citations += f"   Text: \"{chunk['text']}\"\n\n"
            
            web_citations = ""
            if 'web_search_results' in rag_response:
                web_citations = "\n\n**🌐 WEB SEARCH SOURCES USED:**\n"
                for i, result in enumerate(rag_response['web_search_results'][:5], 1):
                    web_citations += f"{i}. **{result.get('title', 'Unknown')}**\n"
                    web_citations += f"   URL: {result.get('url', 'N/A')}\n"
                    web_citations += f"   Content: \"{result.get('snippet', result.get('content', ''))}\"\n\n"
            
            response_text = f"""**🌐 Answer based on web search:**

{answer}

{citations}

{web_citations}

**🔍 Information Source:** Web search processed by AI

**⚠️ Note:** Please verify with healthcare professionals.

{disclaimer}"""
        
        else:
            response_text = f"""**❓ Unable to provide comprehensive answer:**

{answer}

**🔍 Information Source:** Limited information available

{disclaimer}"""
        
        return {
            "response": response_text,
            "agent": "clinical",
            "source": source,
            "chunks_found": rag_response.get('chunks_found', 0),
            "relevant_chunks": rag_response.get('relevant_chunks', []),
            "web_search_results": rag_response.get('web_search_results', []),
            "timestamp": rag_response.get('timestamp')
        }

class MultiAgentSystem:
    """
    Main orchestrator for the multi-agent system
    """
    
    def __init__(self, patients_file: str = None):
        # Initialize components
        self.patient_db = PatientDatabase(patients_file)
        self.interaction_logger = InteractionLogger()
        self.rag_system = ImprovedRAGSystem()
        
        # Initialize agents
        self.receptionist = ReceptionistAgent(self.patient_db, self.interaction_logger)
        self.clinical = ClinicalAgent(self.rag_system, self.interaction_logger)
        
        self.current_agent = "receptionist"
        
        logger.info("Multi-Agent System initialized successfully")
    
    def process_user_input(self, user_input: str) -> Dict[str, Any]:
        """
        Main entry point for processing user input
        
        Args:
            user_input: User's message
            
        Returns:
            Response from appropriate agent
        """
        if self.current_agent == "receptionist":
            response = self.receptionist.process_message(user_input)
            
            # Check if we need to route to clinical agent
            if response.get("route_to") == "clinical":
                self.current_agent = "clinical"
                # Process the original question with clinical agent
                clinical_response = self.clinical.process_medical_query(
                    response["original_question"],
                    response.get("patient_data")
                )
                
                # Add routing information
                clinical_response["routed_from"] = "receptionist"
                clinical_response["patient_context"] = response.get("patient_data") is not None
                
                return clinical_response
            
            return response
        
        elif self.current_agent == "clinical":
            # Clinical agent handles medical queries
            patient_data = getattr(self.receptionist, 'current_patient', None)
            patient_dict = patient_data.__dict__ if patient_data else None
            
            return self.clinical.process_medical_query(user_input, patient_dict)
        
        else:
            # Fallback to receptionist
            self.current_agent = "receptionist"
            return self.receptionist.process_message(user_input)
    
    def reset_conversation(self):
        """Reset conversation state"""
        self.current_agent = "receptionist"
        self.receptionist.conversation_state = "greeting"
        self.receptionist.current_patient = None
        logger.info("Conversation state reset")
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of current conversation state"""
        return {
            "current_agent": self.current_agent,
            "receptionist_state": self.receptionist.conversation_state,
            "current_patient": self.receptionist.current_patient.patient_name if self.receptionist.current_patient else None,
            "total_interactions": len(self.interaction_logger.interactions)
        }


def main():
    """Test the multi-agent system"""
    # Initialize system
    system = MultiAgentSystem()
    
    print("Multi-Agent Post-Discharge Care System")
    print("=" * 50)
    print("Type 'quit' to exit, 'reset' to start over")
    print()
    
    # Start conversation
    initial_response = system.process_user_input("Hello")
    print(f"System: {initial_response['response']}")
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'reset':
            system.reset_conversation()
            print("System: Conversation reset. Hello! How can I help you today?")
            continue
        
        if user_input:
            response = system.process_user_input(user_input)
            print(f"\n{response['agent'].title()} Agent: {response['response']}")
            
            # Show conversation summary
            summary = system.get_conversation_summary()
            print(f"\n[Debug] Current agent: {summary['current_agent']}, Patient: {summary['current_patient']}")


if __name__ == "__main__":
    main()
